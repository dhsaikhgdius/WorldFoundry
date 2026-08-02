import { withBasePath } from '@/lib/site-path';

export type HomeHeroSlide =
  | {
      id: string;
      kind: 'video';
      label: string;
      caption: string;
      src: string;
      poster: string;
    }
  | {
      id: string;
      kind: 'image';
      label: string;
      caption: string;
      src: string;
    };

export const homeHeroSlides: HomeHeroSlide[] = [
  {
    id: 'showcase',
    kind: 'video',
    label: 'World models',
    caption: 'Unified inference and evaluation across video, 3D, and embodied stacks.',
    src: withBasePath('/cover_4x4_hero.mp4') ?? '/cover_4x4_hero.mp4',
    poster: withBasePath('/cover_4x4_hero-poster.webp') ?? '/cover_4x4_hero-poster.webp',
  },
  {
    id: 'hunyuan-voyager',
    kind: 'video',
    label: 'Hunyuan Voyager',
    caption: 'Village-scale world exploration from a single prompt.',
    src:
      withBasePath('/demos/studio/hunyuan-world-voyager-case1.mp4') ??
      '/demos/studio/hunyuan-world-voyager-case1.mp4',
    poster: withBasePath('/images/hero/hunyuan-voyager.webp') ?? '/images/hero/hunyuan-voyager.webp',
  },
  {
    id: 'matrix-game-2',
    kind: 'video',
    label: 'Matrix-Game 2',
    caption: 'Interactive world generation with keyboard and mouse control.',
    src:
      withBasePath('/demos/studio/matrix-game-2-official-universal.mp4') ??
      '/demos/studio/matrix-game-2-official-universal.mp4',
    poster: withBasePath('/images/hero/matrix-game-2.webp') ?? '/images/hero/matrix-game-2.webp',
  },
  {
    id: 'neoverse',
    kind: 'video',
    label: 'NeoVerse',
    caption: 'Embodied tabletop manipulation with camera trajectories.',
    src:
      withBasePath('/demos/studio/neoverse-robot-tabletop.mp4') ??
      '/demos/studio/neoverse-robot-tabletop.mp4',
    poster: withBasePath('/images/hero/neoverse.webp') ?? '/images/hero/neoverse.webp',
  },
];
