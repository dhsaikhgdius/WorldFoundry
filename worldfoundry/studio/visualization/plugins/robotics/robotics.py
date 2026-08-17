from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")
from dataclasses import dataclass
from typing import Any, Dict, Optional

import dlimp as dl
import flax
import gym
import jax
import jax.numpy as jnp
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
import tqdm
import wandb

from octo.utils.gym_wrappers import (
    HistoryWrapper,
    NormalizeProprio,
    RHCWrapper,
    TemporalEnsembleWrapper,
)

BASE_METRIC_KEYS = {
    "mse": ("mse", tuple()),  # What is the MSE
    ####
    #
    # XYZ delta metrics
    #
    ####
    # Angle between true and predicted XYZ delta when moving
    "xyz_angle": (
        "xyz_angle",
        ("moving",),
    ),
    # Did we predict the XYZ delta within 0.5 radians when moving
    "xyz_angle_accuracy": (
        "xyz_angle_accuracy",
        ("moving",),
    ),
    # Did we predict the XYZ delta within 0.5 radians and 50% norm when moving
    "xyz_accuracy": (
        "xyz_accuracy",
        ("moving",),
    ),
    ####
    #
    # Gripper metrics
    #
    ####
    # What % of timesteps (near the actual gripper changes) is the predicted gripper correct?
    "gripping_accuracy": ("gripper_correct", ("gripper_changing",)),
    # Gripper prediction accuracy
    "gripping_accuracy_full": ("gripper_correct", tuple()),
    # The metrics below require propio to compute, uncomment if dataloader returns proprio
    # What is the relative height (in m) that we try to grip at, compared to the data?
    # "grip_height": ("height_to_grip", ("is_first_grip",)),
    # "early_gripped": ("early_gripped", ("is_first_grip",)),
    # What percentage of grips do we attempt early (early = higher than the height gripped at in the data)
    # "early_gripped_height_aware": ("early_gripped_height_aware", ("is_first_grip",)),
    # What timestep do we attempt to grip at (relative to the first timestep we should at)
    # "grip_timestep_early": ("timestep_to_grip", ("is_first_grip",)),
}


BASE_SUB_CONDITIONS = dict()


def run_policy_on_trajectory(policy_fn, traj, *, text_processor=None):
    """
    Args:
        policy_fn: A function that takes in a batch of observations and tasks and returns $n$ sampled actions
            of shape (batch_size, n_samples, action_dim). (n_samples can be arbitrary). policy_fn should be
            willing to take in arbitrary batch sizes (use `batched_apply` to wrap a jitted function)
        traj: A dictionary of trajectory data. Should contain "observations", "actions", and "language_instruction" keys.
        text_processor: A function that takes in a batch of text and returns a batch of tokens.
    """
    len_traj = len(traj["action"])

    tasks = {}
    tasks.update(
        jax.tree_map(
            lambda arr: np.tile(arr[-1][-1], (len_traj, *([1] * (arr.ndim - 2)))),
            traj["observation"],
        )
    )
    if text_processor:
        tasks["language_instruction"] = text_processor.encode(
            [s.decode("utf-8") for s in traj["task"]["language_instruction"]]
        )
        tasks["pad_mask_dict"]["language_instruction"] = np.array(
            [len(s.decode("utf-8")) > 0 for s in traj["task"]["language_instruction"]]
        )

    actions = policy_fn(traj["observation"], tasks)

    horizon = jax.tree_util.tree_leaves(traj["observation"])[0].shape[1]
    return {
        "n": np.array(len_traj),
        "pred_actions_chunk": actions,
        "pred_actions": actions[:, :, 0],  # only use first predicted action
        "actions": traj["action"][:, horizon - 1, 0],  # only use first action
        **(
            {"proprio": traj["observation"]["proprio"][:, horizon - 1]}
            if "proprio" in traj["observation"]
            else {}
        ),
    }


@dataclass
class Visualizer:
    dataset: dl.DLataset
    metric_keys: dict = None
    sub_conditions: dict = None
    freeze_trajs: bool = True  # Use the same trajectories every time
    text_processor: object = None

    def __post_init__(self):
        self.action_proprio_stats = self.dataset.dataset_statistics
        cardinality = self.dataset.cardinality()
        if (
            cardinality == tf.data.INFINITE_CARDINALITY
            or cardinality == tf.data.UNKNOWN_CARDINALITY
        ):
            self.cardinality = float("inf")
        else:
            self.cardinality = cardinality.numpy()
        self.visualized_trajs = False
        self._cached_iterators = {}

    def metrics_for_wandb(
        self,
        infos,
        metric_keys=None,
        sub_conditions=None,
    ):
        """Computes aggregate metrics from a list of trajectory info dictionaries.

        Args:
            infos: Returned from `raw_evaluations`
            metric_keys: A dictionary of metrics to measure.
                k: name of metric (for logging)
                v[0]: name of quantity to measure
                v[1]: names of conditions to mask by
            sub_conditions: A dictionary of sub-conditions to measure. e.g. "when_far" or "when_close"
                k: name of sub-condition (for logging)
                v: names of conditions to mask by
        """

        metric_keys = metric_keys or self.metric_keys or BASE_METRIC_KEYS
        sub_conditions = sub_conditions or self.sub_conditions or BASE_SUB_CONDITIONS

        all_info = {
            k: np.concatenate([info[k] for info in infos])
            for k in infos[0]
            if infos[0][k].ndim > 0
        }

        def masked_mean(quantity_key, *mask_keys):
            mask = np.broadcast_to(
                np.product([all_info[k] for k in mask_keys], axis=0),
                all_info[quantity_key].shape,
            )
            return np.sum(all_info[quantity_key] * mask) / np.sum(mask)

        metrics = {}
        for k, (quantity_key, mask_keys) in metric_keys.items():
            metrics[k] = masked_mean(quantity_key, *mask_keys)
            for sub_condition_name, sub_condition in sub_conditions.items():
                metrics[f"{k}_{sub_condition_name}"] = masked_mean(
                    quantity_key, *mask_keys, *sub_condition
                )
        return metrics

    def visualize_for_wandb(
        self,
        policy_fn,
        max_trajs=1,
        add_images=None,
    ):
        """Returns a dictionary of visualizations to log to wandb.
        Args:
            policy_fn: See `raw_evaluations`
            max_trajs: The maximum number of trajectories to visualize.
            add_images: Whether to add images of the trajectory to the visualization.
        """

        iterator = self.get_iterator(self.dataset, max_trajs)
        visualizations = {}

        for n, traj in tqdm.tqdm(enumerate(iterator), total=max_trajs):
            info = run_policy_on_trajectory(
                policy_fn,
                traj,
                text_processor=self.text_processor,
            )
            info = add_unnormalized_info(info, self.action_proprio_stats)
            info = add_manipulation_metrics(info)

            if "unnorm_proprio" in info:
                plotly_fig = plot_trajectory_actions(**info)
                visualizations[f"traj_{n}"] = plotly_fig

            # plot qualitative action trajectory per dimension w/ and w/o action chunk
            visualizations[f"traj_{n}_mpl"] = plot_trajectory_overview_mpl(
                traj, act=info["unnorm_pred_actions_chunk"][:, :, :1], **info
            )
            visualizations[f"traj_{n}_mpl_chunk"] = plot_trajectory_overview_mpl(
                traj, act=info["unnorm_pred_actions_chunk"], **info
            )
            if add_images or not self.visualized_trajs:
                for key in filter(lambda key: "image" in key, traj["observation"]):
                    images = traj["observation"][key][:, 0]

                    observation_slice = np.concatenate(
                        images[np.linspace(0, len(images) - 1, 5).astype(int)], 1
                    )
                    visualizations[f"traj_{n}_{key}"] = wandb.Image(observation_slice)
        self.visualized_trajs = True
        return visualizations

    def raw_evaluations(
        self,
        policy_fn,
        max_trajs=int(1e6),
    ):
        """Computes accuracy metrics for trajectories in the dataset.

        Args:
            policy_fn: A function that takes in a batch of observations and goals and returns sampled actions
                of shape (batch_size, n_samples, action_dim). (n_samples can be arbitrary)
            max_trajs: The maximum number of trajectories to evaluate on.
        Returns:
            all_traj_info: A list of dictionaries containing information about each trajectory (pass into `process_for_wandb`)
        """
        iterator = self.get_iterator(self.dataset, max_trajs)

        all_traj_info = []

        for traj in tqdm.tqdm(iterator, total=max_trajs):
            info = run_policy_on_trajectory(
                policy_fn,
                traj,
                text_processor=self.text_processor,
            )
            info = add_unnormalized_info(info, self.action_proprio_stats)
            info = add_manipulation_metrics(info)
            all_traj_info.append(info)
        return all_traj_info

    def get_iterator(self, dataset, n):
        n = min(n, self.cardinality)
        if n not in self._cached_iterators:
            self._cached_iterators[n] = (
                dataset.take(n).repeat().as_numpy_iterator()
                if self.freeze_trajs
                else dataset.repeat().as_numpy_iterator()
            )
        return itertools.islice(self._cached_iterators[n], n)


@dataclass
class RolloutVisualizer:
    """
    Runs policy rollouts on a given simulated environment.

    Args:
        env_name (str): Gym.make environment creation string
        history_length (int): Number of history steps policy gets conditioned on (window_size).
        exec_horizon (int): Number of executed action steps.
        max_episode_length (int): Max number of steps per rollout episode.
        env_kwargs (dict): Additional kwargs to pass to gym.make
        use_temp_ensembling (bool): Whether to use temporal ensembling or receding horizon control.
        vis_fps (int): FPS of logged rollout video
        video_subsample_rate (int): Subsampling rate for video logging (to reduce video size for high-frequency control)
        action_proprio_metadata (dict): Dictionary of normalization statistics for proprio and actions.
    """

    name: str
    env_name: str
    history_length: int
    exec_horizon: int
    max_episode_length: int
    env_kwargs: Dict[str, Any]
    use_temp_ensembling: bool = True
    vis_fps: int = 10
    video_subsample_rate: int = 1
    action_proprio_metadata: Optional[dict] = None

    def __post_init__(self):
        self._env = gym.make(self.env_name, **self.env_kwargs)
        if self.action_proprio_metadata is not None:
            self._env = NormalizeProprio(self._env, self.action_proprio_metadata)
        self._env = HistoryWrapper(
            self._env,
            self.history_length,
        )
        if self.use_temp_ensembling:
            self._env = TemporalEnsembleWrapper(self._env, self.exec_horizon)
        else:
            self._env = RHCWrapper(self._env, self.exec_horizon)

    def run_rollouts(self, policy_fn, state, mode, n_rollouts=10, n_vis_rollouts=3):
        def listdict2dictlist(LD):
            return {k: [dic[k] for dic in LD] for k in LD[0]}

        rollout_info = {
            "episode_returns": [],
            "episode_metrics": [],
        }
        for rollout_idx in tqdm.tqdm(range(n_rollouts)):
            obs, info = self._env.reset()
            if mode == "text_conditioned":
                task = state.model.create_tasks(texts=[self._env.get_instruction()])
            elif mode == "image_conditioned":
                task = state.model.create_tasks(goals=self._env.get_goal())
            else:
                raise ValueError(f"Rollout eval mode {mode} not supported")
            images = [obs["image_primary"][-1]]
            episode_return = 0.0
            metrics = []
            while len(images) < self.max_episode_length:
                # policy outputs are shape [batch, n_samples, pred_horizon, act_dim]
                # we remove batch dimension & use first sampled action, ignoring other samples
                actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
                actions = np.array(actions[0, 0])
                obs, reward, done, trunc, info = self._env.step(actions)
                if "observations" in info:
                    images.extend(
                        [o["image_primary"][-1] for o in info["observations"]]
                    )
                else:
                    images.append(obs["image_primary"][-1])
                episode_return += reward
                if "metrics" in info:
                    metrics.append(info["metrics"])
                if done or trunc:
                    break

            rollout_info["episode_returns"].append(episode_return)
            if metrics:
                # concatenate all chunks into one dict of lists, then average across episode
                metrics = listdict2dictlist(metrics)
                rollout_info["episode_metrics"].append(
                    jax.tree_map(lambda x: np.mean(x), metrics)
                )
            if hasattr(self._env, "get_episode_metrics"):
                if metrics:
                    rollout_info["episode_metrics"][-1].update(
                        self._env.get_episode_metrics()
                    )
                else:
                    rollout_info["episode_metrics"].append(
                        self._env.get_episode_metrics()
                    )
            if rollout_idx < n_vis_rollouts:
                # save rollout video
                assert (
                    images[0].dtype == np.uint8
                ), f"Expect uint8, got {images[0].dtype}"
                assert (
                    images[0].shape[-1] == 3
                ), f"Expect [height, width, channels] format, got {images[0].shape}"
                if mode == "image_conditioned":
                    images = [
                        np.concatenate([task["image_primary"][0], frame], axis=0)
                        for frame in images
                    ]
                rollout_info[f"rollout_{rollout_idx}_vid"] = wandb.Video(
                    np.array(images).transpose(0, 3, 1, 2)[
                        :: self.video_subsample_rate
                    ],
                    fps=self.vis_fps,
                )
        rollout_info["avg_return"] = np.mean(rollout_info["episode_returns"])
        rollout_info["episode_returns"] = wandb.Histogram(
            rollout_info["episode_returns"]
        )
        if rollout_info["episode_metrics"]:
            metrics = listdict2dictlist(rollout_info.pop("episode_metrics"))
            for metric in metrics:
                rollout_info[metric] = wandb.Histogram(metrics[metric])
                rollout_info[f"avg_{metric}"] = np.mean(metrics[metric])
        else:
            rollout_info.pop("episode_metrics")
        return rollout_info


def unnormalize(arr, mean, std, mask=None, **kwargs):
    if mask is None:
        mask = np.ones_like(mean)
    adim = mean.shape[0]
    trunc_arr = arr[..., :adim]
    unnorm_arr = np.where(mask, trunc_arr * np.array(std) + np.array(mean), trunc_arr)
    return np.concatenate([unnorm_arr, arr[..., adim:]], axis=-1)


def add_unnormalized_info(
    info,
    normalization_stats,
):
    info.update(
        {
            "unnorm_pred_actions": unnormalize(
                info["pred_actions"], **normalization_stats["action"]
            ),
            "unnorm_pred_actions_chunk": unnormalize(
                info["pred_actions_chunk"], **normalization_stats["action"]
            ),
            "unnorm_actions": unnormalize(
                info["actions"], **normalization_stats["action"]
            ),
            **(
                {
                    "unnorm_proprio": unnormalize(
                        info["proprio"], **normalization_stats["proprio"]
                    )
                }
                if "proprio" in info
                else {}
            ),
        }
    )
    return info


def add_manipulation_metrics(info):
    """Adds metrics to the info dictionary from `run_policy_on_trajectory`

    Assumes the following structure of the actions:
        actions[:, :3] = xyz
        actions[:, 3:6] = xyz rotation
        actions[:, 6] = gripper

    Also assumes that unnormalized actions correspond to deltas (measured in meters) from the previous timestep.
    Also assumes that the gripper is closed when the gripper value is > 0.5

    (Note: these are all defaults in the Bridge dataset)
    """
    multiple_sample_info = {k: v for k, v in info.items() if v.ndim == 3}
    single_sample_info = {k: v for k, v in info.items() if v.ndim != 3}

    def per_sample_info(multi_info, single_info):
        kwargs = {**multi_info, **single_info}
        return {
            **_gripper_info(**kwargs),
            **_mse_info(**kwargs),
            **_xyz_info(**kwargs),
            **_condition_info(**kwargs),
            **(_gripping_early_metrics(**kwargs) if "proprio" in kwargs else {}),
        }

    new_metrics = jax.vmap(per_sample_info, in_axes=(1, None), out_axes=1)(
        multiple_sample_info, single_sample_info
    )
    return flax.core.copy(info, new_metrics)


def plot_trajectory_actions(
    unnorm_pred_actions,
    unnorm_actions,
    unnorm_proprio,
    **kwargs,
):
    """Creates a 3D plotly figure of the trajectory and predicted actions."""
    pred_actions, actions, proprio = unnorm_pred_actions, unnorm_actions, unnorm_proprio

    # TODO: make this less hardcoded
    proprio = np.concatenate(
        [proprio[..., 1:7], proprio[..., -1:]], axis=-1
    )  # extract proprio

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=proprio[:, 0],
            y=proprio[:, 1],
            z=proprio[:, 2],
            marker=dict(
                size=4,
                color=np.arange(len(proprio)),
                colorscale="Viridis",
            ),
            line=dict(color="darkblue", width=2),
        )
    )

    last_plotted = 0
    for i in range(len(actions) - 1):
        visible = np.linalg.norm((proprio[i] - proprio[last_plotted])[:3]) > 0.05
        visible = visible or (i == 0)
        if visible:
            last_plotted = i

        xs = []
        ys = []
        zs = []
        for action in pred_actions[i]:
            ns = proprio[i] + action
            xs.extend((proprio[i, 0], ns[0]))
            ys.extend((proprio[i, 1], ns[1]))
            zs.extend((proprio[i, 2], ns[2]))
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                visible="legendonly" if not visible else True,
                name="timestep {}".format(i),
                marker=dict(size=1, opacity=0),
                line=dict(color="rgba(0, 0, 255, 0.1)"),
            )
        )
    fig.update_layout(
        scene=dict(
            annotations=[
                dict(
                    x=proprio[0, 0],
                    y=proprio[0, 1],
                    z=proprio[0, 2],
                    text="start",
                    xanchor="left",
                    opacity=0.7,
                ),
                dict(x=proprio[-1, 0], y=proprio[-1, 1], z=proprio[-1, 2], text="goal"),
            ]
        )
    )
    return fig


class WandBFigure:
    def __init__(self, save_to=None, **figure_kwargs):
        self.fig = plt.figure(**figure_kwargs)
        self.canvas = FigureCanvas(self.fig)

    def __enter__(self):
        return plt.figure(self.fig.number)

    def __exit__(self, exc_type, exc_value, traceback):
        self.canvas.draw()
        out_image = np.frombuffer(self.canvas.tostring_rgb(), dtype="uint8")
        self.image = out_image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(self.fig)


def plot_trajectory_overview_mpl(
    traj,
    act,
    unnorm_actions,
    **info,
):
    n_act_dims = traj["action"].shape[-1]
    grid_size = int(np.ceil(np.sqrt(n_act_dims + 1)))
    wandb_figure = WandBFigure(figsize=(grid_size * 5, grid_size * 5))
    gs = gridspec.GridSpec(grid_size, grid_size)
    with wandb_figure as fig:
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(info["mse"].mean(axis=1))
        ax.set_ylabel("MSE")
        for i in range(n_act_dims):
            ax = fig.add_subplot(gs[(i + 1) // grid_size, (i + 1) % grid_size])
            ax.plot(unnorm_actions[:, i], label="action")
            # plot predicted action chunks, act.shape = [time, n_samples, chunk, act_dim]
            chunk_length = act.shape[2]
            for t in range(act.shape[0]):
                step_idx, chunk_idx = divmod(t, chunk_length)
                unnorm_pred_actions_i = act[
                    int(step_idx * chunk_length), :, chunk_idx, i
                ]
                x = np.full((unnorm_pred_actions_i.shape[0],), t)
                ax.scatter(
                    x.flat[:],
                    unnorm_pred_actions_i.flat[:],
                    color="tab:red",
                    s=4,
                    alpha=0.5,
                )
                if chunk_idx == 0 and (act.shape[0] // chunk_length) <= 20:
                    ax.axvline(t, color="red", linestyle="--", alpha=0.2)
            ax.set_ylabel(f"dim {i}")
        fig.suptitle(traj["task"]["language_instruction"][0].decode("utf-8"))
    return wandb.Image(wandb_figure.image)


#############################################
#
#
#   A list of metrics to compute on the trajectory
#
#
#############################################


def _get_gripper(actions):
    return actions[:, -1]  # Hard-coded


def _get_xyz(actions):
    return actions[:, :3]  # Hard-coded


def _gripper_closed(actions):
    return _get_gripper(actions) > 0.5  # Hard-coded


def _gripper_correct(unnorm_actions, unnorm_pred_actions, **kwargs):
    return jnp.equal(
        _gripper_closed(unnorm_actions), _gripper_closed(unnorm_pred_actions)
    )


def _xyz_angle(unnorm_actions, unnorm_pred_actions, **kwargs):
    def angle_between(v1, v2):
        v1_u = v1 / (1e-6 + jnp.linalg.norm(v1))
        v2_u = v2 / (1e-6 + jnp.linalg.norm(v2))
        return jnp.arccos(jnp.clip(jnp.dot(v1_u, v2_u), -1.0, 1.0))

    return jax.vmap(angle_between)(
        _get_xyz(unnorm_actions), _get_xyz(unnorm_pred_actions)
    )


def _xyz_close(unnorm_actions, unnorm_pred_actions, **kwargs):
    norm1 = jnp.linalg.norm(_get_xyz(unnorm_actions), axis=-1)
    norm2 = jnp.linalg.norm(_get_xyz(unnorm_pred_actions), axis=-1)
    angle = _xyz_angle(
        unnorm_actions=unnorm_actions, unnorm_pred_actions=unnorm_pred_actions
    )
    return jnp.logical_and(
        angle < 0.5,
        (norm1 > 0.5 * norm2) & (norm2 > 0.5 * norm1),
    )


def _mse(actions, pred_actions, dims=None, **kwargs):
    # Note: this is the MSE of the normalized actions (not the unnormalized actions)
    delta = actions - pred_actions
    if dims is not None:
        delta = delta[:, dims]
    return jnp.sum(delta**2, axis=-1)


def _moving(unnorm_actions, axis=None, magnitude=0, **kwargs):
    if axis is None:
        dist = np.linalg.norm(unnorm_actions[:, :3], axis=1)
    else:
        dist = np.abs(unnorm_actions[:, axis])
    return np.greater(dist, magnitude)


def _xyz_info(**kwargs):
    angle = _xyz_angle(**kwargs)
    return {
        "xyz_angle": angle,
        "xyz_angle_accuracy": angle < 0.5,
        "xyz_accuracy": _xyz_close(**kwargs),
    }


def _mse_info(**kwargs):
    return {
        "mse": _mse(**kwargs),
        "mse_xyz": _mse(dims=[0, 1, 2], **kwargs),  # hard-coded
        "mse_gripper": _mse(dims=[6], **kwargs),  # hard-coded
        "mse_xyzrotation": _mse(dims=[3, 4, 5], **kwargs),  # hard-coded
    }


def _gripping_early_metrics(
    unnorm_actions, unnorm_proprio, unnorm_pred_actions, **kwargs
):
    gripper_closed = _gripper_closed(unnorm_actions)
    pred_gripper_closed = _gripper_closed(unnorm_pred_actions)

    unnorm_proprio = unnorm_proprio[:, 1:]  # Remove special dimension
    z_position = unnorm_proprio[:, 2]

    first_grip = jnp.logical_and(
        gripper_closed, jnp.logical_not(jnp.roll(gripper_closed, 1, axis=0))
    )  # Was the gripper closed at the last timestep?

    gripped_i_steps_early = {
        i: jnp.logical_and(
            first_grip,
            jnp.roll(pred_gripper_closed, i, axis=0),  # Predicted a grip i steps early
        )
        for i in range(1, 5)
    }
    early_gripped = sum(gripped_i_steps_early.values()) > 0

    gripped_i_steps_early_height_aware = {
        i: jnp.logical_and(
            gripped_i_steps_early[i],
            jnp.roll(z_position, i, axis=0) - z_position > 0.005,
        )
        for i in range(1, 5)
    }  # also check that the z position increased
    early_gripped_height_aware = sum(gripped_i_steps_early_height_aware.values()) > 0

    height_to_grip = jnp.zeros_like(z_position)
    timestep_to_grip = jnp.zeros_like(z_position)
    for i in range(1, 5):
        new_height_to_grip = jnp.where(
            jnp.roll(pred_gripper_closed, i, axis=0),
            jnp.roll(z_position, i, axis=0) - z_position,
            0,
        )
        height_to_grip = jnp.maximum(height_to_grip, new_height_to_grip)
        timestep_to_grip = jnp.maximum(
            timestep_to_grip,
            jnp.where(
                jnp.roll(pred_gripper_closed, i, axis=0),
                i,
                0,
            ),
        )
    height_to_grip = jnp.where(first_grip, height_to_grip, 0)
    timestep_to_grip = jnp.where(first_grip, timestep_to_grip, 0)

    gripped_within_two_steps = jnp.logical_and(
        first_grip,
        jnp.logical_or(
            pred_gripper_closed,  # Predicted at this timestep
            jnp.roll(
                pred_gripper_closed, -1, axis=0
            ),  # Predicted at the next timestep. Note that the image of the gripper may already be closed, so this might not be a very reliable metric
        ),
    )
    return {
        "is_first_grip": first_grip,
        "height_to_grip": height_to_grip,
        "early_gripped": early_gripped,
        "early_gripped_height_aware": early_gripped_height_aware,
        "timestep_to_grip": timestep_to_grip,
        "gripped_on_time": gripped_within_two_steps,
    }


def _gripper_info(**kwargs):
    gripper_correct = _gripper_correct(**kwargs)

    actions = kwargs.get("unnorm_actions")
    past_actions = jnp.roll(actions, 3, axis=0)
    future_actions = jnp.roll(actions, -3, axis=0)
    gripping = jnp.logical_or(
        jnp.logical_and(
            _gripper_closed(actions), jnp.logical_not(_gripper_closed(past_actions))
        ),  # Gripper was open in the past, but is closed now
        jnp.logical_and(
            _gripper_closed(future_actions), jnp.logical_not(_gripper_closed(actions))
        ),  # Gripper is open now, but will be closed in the future
    )

    releasing = jnp.logical_or(
        jnp.logical_and(
            _gripper_closed(past_actions), jnp.logical_not(_gripper_closed(actions))
        ),  # Gripper was closed in the past, but is open now
        jnp.logical_and(
            _gripper_closed(actions), jnp.logical_not(_gripper_closed(future_actions))
        ),  # Gripper is closed now, but will be open in the future
    )

    gripper_changing = jnp.logical_or(gripping, releasing)
    still = jnp.logical_not(gripper_changing)
    return {
        "gripper_correct": gripper_correct,
        "gripping": gripping,
        "releasing": releasing,
        "still": still,
        "gripper_changing": gripper_changing,
    }


def _condition_info(**kwargs):
    actions, n = kwargs.get("unnorm_actions"), kwargs.get("n")
    distance = n - np.arange(len(actions))
    return {
        "<10_to_end": distance < 10,
        ">20_to_end": distance > 20,
        "moving": _moving(**kwargs, magnitude=0.01),  # Moved at least 1cm (hard-coded)
    }


"""Debug plots for LeRobot async inference action queues."""


import matplotlib.pyplot as plt


def visualize_action_queue_size(timestamps: list[float], queue_sizes: list[int]) -> None:
    """Plot action queue size over time for debugging."""
    _, ax = plt.subplots()
    ax.set_title("Action Queue Size Over Time")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Action Queue Size")
    if queue_sizes:
        ax.set_ylim(0, max(queue_sizes) * 1.1)
    ax.grid(True, alpha=0.3)
    ax.plot(timestamps, queue_sizes)
    plt.show()


#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualization utilities for RTC debug information."""

import torch


class RTCDebugVisualizer:
    """Visualizer for RTC debug information.

    This class provides methods to visualize debug information collected by the Tracker,
    including corrections, errors, weights, and guidance weights over denoising steps.
    """

    @staticmethod
    def plot_waypoints(
        axes,
        tensor,
        start_from: int = 0,
        color: str = "blue",
        label: str = "",
        alpha: float = 0.7,
        linewidth: float = 2,
        marker: str | None = None,
        markersize: int = 4,
    ):
        """Plot trajectories across multiple dimensions.

        This function plots a tensor's values across time for multiple dimensions,
        with each dimension plotted on a separate axis.

        Args:
            axes: Array of matplotlib axes (one for each dimension).
            tensor: The tensor to plot (can be torch.Tensor or numpy array).
                   Shape should be (time_steps, num_dims) or (batch, time_steps, num_dims).
            start_from: Starting index for the x-axis.
            color: Color for the plot lines.
            label: Label for the plot legend.
            alpha: Transparency level for the plot.
            linewidth: Width of the plot lines.
            marker: Marker style for data points (e.g., 'o', 's', '^').
            markersize: Size of the markers.
        """
        import numpy as np

        # Handle None tensor
        if tensor is None:
            return

        # Convert tensor to numpy if needed
        tensor_np = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor

        # Handle different tensor shapes
        if tensor_np.ndim == 3:
            # If batch dimension present, take first batch
            tensor_np = tensor_np[0]
        elif tensor_np.ndim == 1:
            # If 1D, reshape to (time_steps, 1)
            tensor_np = tensor_np.reshape(-1, 1)

        # Get dimensions
        time_steps, num_dims = tensor_np.shape

        # Create x-axis indices
        x_indices = np.arange(start_from, start_from + time_steps)

        # Plot each dimension on its corresponding axis
        num_axes = len(axes) if hasattr(axes, "__len__") else 1
        for dim_idx in range(min(num_dims, num_axes)):
            ax = axes[dim_idx] if hasattr(axes, "__len__") else axes

            # Plot the trajectory
            if marker:
                ax.plot(
                    x_indices,
                    tensor_np[:, dim_idx],
                    color=color,
                    label=label if dim_idx == 0 else "",  # Only show label once
                    alpha=alpha,
                    linewidth=linewidth,
                    marker=marker,
                    markersize=markersize,
                )
            else:
                ax.plot(
                    x_indices,
                    tensor_np[:, dim_idx],
                    color=color,
                    label=label if dim_idx == 0 else "",  # Only show label once
                    alpha=alpha,
                    linewidth=linewidth,
                )

            # Add grid and labels if not already present
            if not ax.xaxis.get_label().get_text():
                ax.set_xlabel("Step", fontsize=10)
            if not ax.yaxis.get_label().get_text():
                ax.set_ylabel(f"Dim {dim_idx}", fontsize=10)
            ax.grid(True, alpha=0.3)


#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module handles calibration of hall effect sensors used in the exoskeleton.
Each joint has a pair of ADC channels outputting sin and cos values that trace an ellipse
as the joint rotates due to imprecision in magnet/sensor placement. We fit this ellipse to a unit circle,
and calculate arctan2 of the unit circle to get the joint angle.
We then store the ellipse parameters and the zero offset for each joint to be used at runtime.
"""


import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _serial_available, require_package

if TYPE_CHECKING or _serial_available:
    import serial
else:
    serial = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


ADC_MAX = 2**12 - 1
ADC_HALF = ADC_MAX / 2

# exoskeleton joint names -> ADC channel pairs. TODO: add wrist pitch and wrist yaw
JOINTS = {
    "shoulder_pitch": (0, 1),
    "shoulder_yaw": (2, 3),
    "shoulder_roll": (4, 5),
    "elbow_flex": (6, 7),
    "wrist_roll": (14, 15),
}


@dataclass
class ExoskeletonJointCalibration:
    name: str  # joint name
    center_fit: list[float]  # center of the ellipse
    T: list[list[float]]  # 2x2 transformation matrix
    zero_offset: float = 0.0  # angle at neutral pose


@dataclass
class ExoskeletonCalibration:
    """Full calibration data for an exoskeleton arm."""

    version: int = 2
    side: str = ""
    adc_max: int = ADC_MAX
    joints: list[ExoskeletonJointCalibration] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "side": self.side,
            "adc_max": self.adc_max,
            "joints": [
                {
                    "name": j.name,
                    "center_fit": j.center_fit,
                    "T": j.T,
                    "zero_offset": j.zero_offset,
                }
                for j in self.joints
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExoskeletonCalibration:
        joints = [
            ExoskeletonJointCalibration(
                name=j["name"],
                center_fit=j["center_fit"],
                T=j["T"],
                zero_offset=j.get("zero_offset", 0.0),
            )
            for j in data.get("joints", [])
        ]
        return cls(
            version=data.get("version", 2),
            side=data.get("side", ""),
            adc_max=data.get("adc_max", ADC_MAX),
            joints=joints,
        )


@dataclass(frozen=True)
class CalibParams:
    fit_every: float = 0.15
    min_fit_points: int = 60
    fit_window: int = 900
    max_fit_points: int = 300
    trim_low: float = 0.05
    trim_high: float = 0.95
    median_window: int = 5
    history: int = 3500
    draw_hz: float = 120.0
    sample_count: int = 50


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def joint_z_and_angle(raw16: list[int], j: ExoskeletonJointCalibration) -> tuple[np.ndarray, float]:
    """
    Applies calibration to each joint: raw → centered → ellipse-to-circle → angle.
    """
    pair = JOINTS[j.name]
    s, c = raw16[pair[0]], raw16[pair[1]]  # get sin and cos
    p = np.array([float(c) - ADC_HALF, float(s) - ADC_HALF])  # center the raw values
    z = np.asarray(j.T) @ (
        p - np.asarray(j.center_fit)
    )  # center the ellipse and invert the transformation matrix to get unit circle coords
    ang = float(np.arctan2(z[1], z[0])) - j.zero_offset  # calculate the anvgle and apply the zero offset
    return z, normalize_angle(-ang)  # ensure range is [-pi, pi]


def exo_raw_to_angles(raw16: list[int], calib: ExoskeletonCalibration) -> dict[str, float]:
    """Convert raw sensor readings to joint angles using calibration."""
    return {j.name: joint_z_and_angle(raw16, j)[1] for j in calib.joints}


def run_exo_calibration(
    ser: serial.Serial,
    side: str,
    save_path: Path,
    params: CalibParams | None = None,
) -> ExoskeletonCalibration:
    """
    Run interactive calibration for an exoskeleton arm.
    """
    require_package("pyserial", extra="unitree_g1", import_name="serial")
    try:
        import cv2
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Calibration requires matplotlib and opencv-python. "
            "Install with: pip install matplotlib opencv-python"
        ) from e

    from .exo_serial import read_raw_from_serial

    params = params or CalibParams()
    joint_list = list(JOINTS.items())  # Convert dict to list for indexing
    logger.info(f"Starting calibration for {side} exoskeleton arm")

    def running_median(win: deque) -> float:
        return float(np.median(np.fromiter(win, dtype=float)))

    def read_joint_point(raw16: list[int], pair: tuple[int, int]):
        s, c = raw16[pair[0]], raw16[pair[1]]
        return float(c) - ADC_HALF, float(s) - ADC_HALF, float(s), float(c)

    def select_fit_subset(xs, ys):
        """Select and filter points for ellipse fitting. Trims outliers by radius and downsamples."""
        n = min(params.fit_window, len(xs))
        if n <= 0:
            return None, None
        x = np.asarray(list(xs)[-n:], dtype=float)  # most recent n samples
        y = np.asarray(list(ys)[-n:], dtype=float)
        r = np.sqrt(x * x + y * y)  # radius from origin
        if len(r) >= 20:
            lo, hi = np.quantile(r, params.trim_low), np.quantile(r, params.trim_high)  # outlier bounds
            keep = (r >= lo) & (r <= hi)
            x, y = x[keep], y[keep]  # remove outliers
        if len(x) > params.max_fit_points:
            idx = np.linspace(0, len(x) - 1, params.max_fit_points).astype(int)  # downsample evenly
            x, y = x[idx], y[idx]
        return x, y

    def fit_ellipse_opencv(x, y):
        """Fit ellipse to (x,y) points using OpenCV. Returns center, axes, rotation matrix, and outline."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if len(x) < 5:
            return None
        pts = np.stack([x, y], axis=1).astype(np.float32).reshape(-1, 1, 2)
        try:
            (xc, yc), (w, h), angle_deg = cv2.fitEllipse(pts)  # returns center, axes, rotation in degrees
        except cv2.error:
            return None
        a, b = float(w) * 0.5, float(h) * 0.5  # get ellipse major and minor semi-axes
        phi = np.deg2rad(float(angle_deg))  # to rad
        if b > a:  # ensure major axis is a
            a, b = b, a
            phi += np.pi / 2.0
        if not np.isfinite(a) or not np.isfinite(b) or a <= 1e-6 or b <= 1e-6:
            return None
        cp, sp = float(np.cos(phi)), float(np.sin(phi))  #
        rot = np.array([[cp, -sp], [sp, cp]], dtype=float)  # 2x2 rotation matrix
        center = np.array([float(xc), float(yc)], dtype=float)  # offset vector
        tt = np.linspace(0, 2 * np.pi, 360)
        outline = (rot @ np.stack([a * np.cos(tt), b * np.sin(tt)])).T + center  # for viz
        return {"center": center, "a": a, "b": b, "R": rot, "ex": outline[:, 0], "ey": outline[:, 1]}

    # Setup matplotlib
    plt.ion()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    ax0.set_xlabel("cos - center")
    ax0.set_ylabel("sin - center")
    ax0.grid(True, alpha=0.25)
    ax0.set_aspect("equal", adjustable="box")
    ax1.set_title("Unit circle + angle")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True, alpha=0.25)
    ax1.set_aspect("equal", adjustable="box")
    tt = np.linspace(0, 2 * np.pi, 360)
    ax1.plot(np.cos(tt), np.sin(tt), "k-", linewidth=1)
    ax0.set_xlim(-2200, 2200)
    ax0.set_ylim(-2200, 2200)
    ax1.set_xlim(-1.4, 1.4)
    ax1.set_ylim(-1.4, 1.4)

    sc0 = ax0.scatter([], [], s=6, animated=True)
    (ell_line,) = ax0.plot([], [], "r-", linewidth=2, animated=True)
    sc1 = ax1.scatter([], [], s=6, animated=True)
    (radius_line,) = ax1.plot([], [], "g-", linewidth=2, animated=True)
    angle_text = ax1.text(
        0.02, 0.98, "", transform=ax1.transAxes, va="top", ha="left", fontsize=12, animated=True
    )

    fig.canvas.draw()
    bg0 = fig.canvas.copy_from_bbox(ax0.bbox)
    bg1 = fig.canvas.copy_from_bbox(ax1.bbox)

    # State
    joints_out = []
    joint_idx = 0
    phase = "ellipse"
    advance_requested = False
    zero_samples = []

    def on_key(event):
        nonlocal advance_requested
        if event.key in ("n", "N", "enter", " "):
            advance_requested = True

    fig.canvas.mpl_connect("key_press_event", on_key)

    def reset_state():
        return {
            "xs": deque(maxlen=params.history),
            "ys": deque(maxlen=params.history),
            "xu": deque(maxlen=params.history),
            "yu": deque(maxlen=params.history),
            "win_s": deque(maxlen=params.median_window),
            "win_c": deque(maxlen=params.median_window),
            "ellipse_cache": None,
            "T": None,
            "center_fit": None,
            "have_transform": False,
            "latest_z": None,
            "last_fit": 0.0,
        }

    state = reset_state()
    last_draw = 0.0
    name, pair = joint_list[joint_idx]
    fig.canvas.manager.set_window_title(f"[{joint_idx + 1}/{len(joint_list)}] {name} - ELLIPSE")
    ax0.set_title(f"{name} raw (filtered)")
    logger.info(f"[{joint_idx + 1}/{len(joint_list)}] Calibrating {name}")
    logger.info("Step 1: Move joint around to map ellipse, then press 'n'")

    try:
        while plt.fignum_exists(fig.number):
            name, pair = joint_list[joint_idx]

            # Handles calibration GUI state: ellipse → zero_pose → next joint -> ellipse -> ...
            if phase == "ellipse" and advance_requested and state["have_transform"]:
                joints_out.append(
                    {
                        "name": name,
                        "center_fit": state["center_fit"].tolist(),
                        "T": state["T"].tolist(),
                    }
                )
                logger.info(f"  -> Ellipse saved for {name}")
                phase, zero_samples, advance_requested = "zero_pose", [], False
                fig.canvas.manager.set_window_title(f"[{joint_idx + 1}/{len(joint_list)}] {name} - ZERO POSE")
                ax0.set_title(f"{name} - hold zero pose")
                fig.canvas.draw()
                bg0, bg1 = fig.canvas.copy_from_bbox(ax0.bbox), fig.canvas.copy_from_bbox(ax1.bbox)
                logger.info(f"Step 2: Hold {name} in zero position, then press 'n'")

            elif phase == "ellipse" and advance_requested and not state["have_transform"]:
                logger.info("  (Need valid fit first - keep moving the joint)")
                advance_requested = False

            elif phase == "zero_pose" and advance_requested:
                if len(zero_samples) >= params.sample_count:
                    zero_offset = float(np.mean(zero_samples[-params.sample_count :]))
                    joints_out[-1]["zero_offset"] = zero_offset
                    logger.info(f"  -> {name} zero: {zero_offset:+.3f} rad ({np.degrees(zero_offset):+.1f}°)")
                    joint_idx += 1
                    advance_requested = False

                    if joint_idx >= len(joint_list):
                        # All joints done
                        calib = ExoskeletonCalibration(
                            version=2,
                            side=side,
                            adc_max=ADC_MAX,
                            joints=[
                                ExoskeletonJointCalibration(
                                    name=j["name"],
                                    center_fit=j["center_fit"],
                                    T=j["T"],
                                    zero_offset=j.get("zero_offset", 0.0),
                                )
                                for j in joints_out
                            ],
                        )
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(save_path, "w") as f:
                            json.dump(calib.to_dict(), f, indent=2)
                        logger.info(f"Saved calibration to {save_path}")
                        logger.info("Calibration complete!")
                        plt.close(fig)
                        return calib

                    # Next joint
                    phase, state = "ellipse", reset_state()
                    name, pair = joint_list[joint_idx]
                    fig.canvas.manager.set_window_title(
                        f"[{joint_idx + 1}/{len(joint_list)}] {name} - ELLIPSE"
                    )
                    ax0.set_title(f"{name} raw (filtered)")
                    fig.canvas.draw()
                    bg0, bg1 = fig.canvas.copy_from_bbox(ax0.bbox), fig.canvas.copy_from_bbox(ax1.bbox)
                    logger.info(f"[{joint_idx + 1}/{len(joint_list)}] Calibrating {name}")
                    logger.info("Step 1: Move joint around to map ellipse, then press 'n'")
                else:
                    logger.info(
                        f"  (Collecting samples: {len(zero_samples)}/{params.sample_count} - hold still)"
                    )
                    advance_requested = False

            # Read sensor
            raw16 = read_raw_from_serial(ser)
            if raw16 is not None:
                x_raw, y_raw, s_raw, c_raw = read_joint_point(raw16, pair)

                if phase == "ellipse":
                    if state["have_transform"]:
                        z = state["T"] @ (np.array([x_raw, y_raw]) - state["center_fit"])
                        state["xu"].append(float(z[0]))
                        state["yu"].append(float(z[1]))
                        state["latest_z"] = (float(z[0]), float(z[1]))
                    state["win_s"].append(s_raw)
                    state["win_c"].append(c_raw)
                    if len(state["win_s"]) >= max(3, params.median_window):
                        state["ys"].append(running_median(state["win_s"]) - ADC_HALF)
                        state["xs"].append(running_median(state["win_c"]) - ADC_HALF)
                else:
                    jdata = joints_out[-1]
                    z = np.array(jdata["T"]) @ (np.array([x_raw, y_raw]) - np.array(jdata["center_fit"]))
                    zero_samples.append(float(np.arctan2(z[1], z[0])))
                    state["latest_z"] = (float(z[0]), float(z[1]))

            # Ellipse fitting
            t = time.time()
            if (
                phase == "ellipse"
                and (t - state["last_fit"]) >= params.fit_every
                and len(state["xs"]) >= params.min_fit_points
            ):
                xfit, yfit = select_fit_subset(state["xs"], state["ys"])
                if xfit is not None and len(xfit) >= params.min_fit_points:
                    fit = fit_ellipse_opencv(xfit, yfit)
                    if fit is not None:
                        state["center_fit"] = fit["center"]
                        state["T"] = np.diag([1.0 / fit["a"], 1.0 / fit["b"]]) @ fit["R"].T
                        state["ellipse_cache"] = (fit["ex"], fit["ey"])
                        state["have_transform"] = True
                state["last_fit"] = t

            # Drawing
            if (t - last_draw) >= 1.0 / params.draw_hz:
                fig.canvas.restore_region(bg0)
                fig.canvas.restore_region(bg1)

                if phase == "ellipse":
                    sc0.set_offsets(np.c_[state["xs"], state["ys"]] if state["xs"] else np.empty((0, 2)))
                    ax0.draw_artist(sc0)
                    ell_line.set_data(*state["ellipse_cache"] if state["ellipse_cache"] else ([], []))
                    ax0.draw_artist(ell_line)
                    sc1.set_offsets(np.c_[state["xu"], state["yu"]] if state["xu"] else np.empty((0, 2)))
                    ax1.draw_artist(sc1)
                    if state["latest_z"]:
                        zx, zy = state["latest_z"]
                        radius_line.set_data([0.0, zx], [0.0, zy])
                        ang = float(np.arctan2(zy, zx))
                        angle_text.set_text(
                            f"angle: {ang:+.3f} rad  ({np.degrees(ang):+.1f}°)\nmove {name}, press 'n' to advance"
                        )
                    else:
                        radius_line.set_data([], [])
                        angle_text.set_text("(waiting for fit)")
                else:
                    sc0.set_offsets(np.empty((0, 2)))
                    ax0.draw_artist(sc0)
                    ell_line.set_data([], [])
                    ax0.draw_artist(ell_line)
                    if state["latest_z"]:
                        zx, zy = state["latest_z"]
                        sc1.set_offsets([[zx, zy]])
                        radius_line.set_data([0.0, zx], [0.0, zy])
                        ang = float(np.arctan2(zy, zx))
                        angle_text.set_text(
                            f"Zero pose for {name}\nangle: {ang:+.3f} rad\nsamples: {len(zero_samples)}/{params.sample_count}\nhold still, press 'n'"
                        )
                    else:
                        sc1.set_offsets(np.empty((0, 2)))
                        radius_line.set_data([], [])
                        angle_text.set_text("(waiting for data)")
                    ax1.draw_artist(sc1)

                ax1.draw_artist(radius_line)
                ax1.draw_artist(angle_text)
                fig.canvas.blit(ax0.bbox)
                fig.canvas.blit(ax1.bbox)
                fig.canvas.flush_events()
                last_draw = t

            plt.pause(0.001)

    finally:
        plt.close(fig)


# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numbers
import os

import numpy as np

from lerobot.types import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR
from .import_utils import require_package


def init_rerun(
    session_name: str = "lerobot_control_loop", ip: str | None = None, port: int | None = None
) -> None:
    """
    Initializes the Rerun SDK for visualizing the control loop.

    Args:
        session_name: Name of the Rerun session.
        ip: Optional IP for connecting to a Rerun server.
        port: Optional port for connecting to a Rerun server.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    if ip and port:
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
    else:
        rr.spawn(memory_limit=memory_limit)


def shutdown_rerun() -> None:
    """Shuts down the Rerun SDK gracefully."""

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    rr.rerun_shutdown()


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format, (optionally) compressed to JPEG and logged as `rr.Image` or `rr.EncodedImage`.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Whether to compress images before logging to save bandwidth & memory in exchange for cpu and quality.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    rr.log(key, entity=img_entity, static=True)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
