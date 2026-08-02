'use client';

import { ImageZoom } from 'fumadocs-ui/components/image-zoom';
import type { ComponentPropsWithoutRef } from 'react';

import { withBasePath } from '@/lib/site-path';

type StaticImageDataLike = {
  src: string;
  height?: number;
  width?: number;
};

type DocsZoomImageProps = Omit<ComponentPropsWithoutRef<'img'>, 'src'> & {
  src?: string | StaticImageDataLike;
};

function resolveSrc(src: DocsZoomImageProps['src']) {
  if (!src) return undefined;
  if (typeof src === 'string') return withBasePath(src) ?? src;
  return withBasePath(src.src) ?? src.src;
}

function mergeClassName(className?: string) {
  return className ? `wf-docs-zoom-image ${className}` : 'wf-docs-zoom-image';
}

export function DocsZoomImage({
  src,
  alt,
  className,
  width,
  height,
  ...props
}: DocsZoomImageProps) {
  const resolved = resolveSrc(src);
  if (!resolved) return null;

  const resolvedWidth =
    width ?? (src && typeof src === 'object' ? src.width : undefined);
  const resolvedHeight =
    height ?? (src && typeof src === 'object' ? src.height : undefined);
  const mergedClassName = mergeClassName(className);

  if (resolvedWidth != null && resolvedHeight != null) {
    return (
      <ImageZoom
        {...props}
        alt={alt ?? ''}
        className={mergedClassName}
        height={resolvedHeight}
        src={resolved}
        width={resolvedWidth}
      />
    );
  }

  return (
    <ImageZoom zoomInProps={{ src: resolved }}>
      <img {...props} alt={alt ?? ''} className={mergedClassName} src={resolved} />
    </ImageZoom>
  );
}
