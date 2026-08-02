import * as CalloutComponents from 'fumadocs-ui/components/callout';
import * as CardComponents from 'fumadocs-ui/components/card';
import * as AccordionComponents from 'fumadocs-ui/components/accordion';
import * as FilesComponents from 'fumadocs-ui/components/files';
import * as StepsComponents from 'fumadocs-ui/components/steps';
import { Banner } from 'fumadocs-ui/components/banner';
import { DynamicCodeBlock } from 'fumadocs-ui/components/dynamic-codeblock';
import { InlineTOC } from 'fumadocs-ui/components/inline-toc';
import { Tab, Tabs } from '@/components/docs-tabs';
import * as TabsComponents from 'fumadocs-ui/components/tabs';
import defaultMdxComponents from 'fumadocs-ui/mdx';
import { DocsGithubInfo } from '@/components/docs-github-info';
import { DocsZoomImage } from '@/components/docs-zoom-image';
import { DocsWelcomeAcknowledgements } from '@/components/docs-welcome-acknowledgements';
import { BenchmarkRecipeCatalog } from '@/components/benchmark-recipe-catalog';
import { CallChainDiagram } from '@/components/call-chain-diagram';
import { MetricQuickNav } from '@/components/metric-quick-nav';
import { ModelRecipeCatalog } from '@/components/model-recipe-catalog';
import { ModelRecipeHeader } from '@/components/model-recipe-header';
import {
  PythonApiCatalog,
  PythonApiGroupReference,
  PythonApiReference,
} from '@/components/python-api-reference';
import { StudioVisualizerGallery } from '@/components/studio-visualizer-gallery';
import { TeaserImage } from '@/components/teaser-image';
import {
  WorldFoundryArchitecture,
  WorldFoundryWorkflow,
} from '@/components/worldfoundry-system-map';
import { withBasePath, withMediaPath } from '@/lib/site-path';
import { TypeTable } from 'fumadocs-ui/components/type-table';
import type { MDXComponents } from 'mdx/types';
import type { ComponentPropsWithoutRef } from 'react';

type StaticImageDataLike = {
  src: string;
  height?: number;
  width?: number;
  blurDataURL?: string;
};

type ImgSrc = ComponentPropsWithoutRef<'img'>['src'];

type DocsImageProps = Omit<ComponentPropsWithoutRef<'img'>, 'src'> & {
  src?: ImgSrc | StaticImageDataLike;
};

function isStaticImageDataLike(src: DocsImageProps['src']): src is StaticImageDataLike {
  return typeof src === 'object' && src !== null && 'src' in src && typeof src.src === 'string';
}

function resolveImageSrc(src: DocsImageProps['src']) {
  if (!src) return src;
  if (typeof src === 'string') return withBasePath(src);
  if (isStaticImageDataLike(src)) {
    return withBasePath(src.src);
  }
  return src;
}

function DocsImage({ src, alt, ...props }: DocsImageProps) {
  const resolved = resolveImageSrc(src);
  const dimensions =
    isStaticImageDataLike(src)
      ? {
          width: props.width ?? src.width,
          height: props.height ?? src.height,
        }
      : {};

  if (typeof resolved !== 'string') {
    return <img {...props} {...dimensions} alt={alt} src={resolved} />;
  }

  return <DocsZoomImage {...props} {...dimensions} alt={alt} src={resolved} />;
}

function DocsVideo({ src, ...props }: ComponentPropsWithoutRef<'video'>) {
  return <video {...props} src={typeof src === 'string' ? withMediaPath(src) : src} />;
}

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...CalloutComponents,
    ...CardComponents,
    ...defaultMdxComponents,
    ...AccordionComponents,
    ...FilesComponents,
    ...StepsComponents,
    ...TabsComponents,
    Banner,
    DynamicCodeBlock,
    InlineTOC,
    Tab,
    Tabs,
    BenchmarkRecipeCatalog,
    CallChainDiagram,
    DocsGithubInfo,
    DocsWelcomeAcknowledgements,
    GithubInfo: DocsGithubInfo,
    MetricQuickNav,
    ModelRecipeCatalog,
    ModelRecipeHeader,
    PythonApiCatalog,
    PythonApiGroupReference,
    PythonApiReference,
    img: DocsImage,
    StudioVisualizerGallery,
    TeaserImage,
    TypeTable,
    Video: DocsVideo,
    WorldFoundryArchitecture,
    WorldFoundryWorkflow,
    video: DocsVideo,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
