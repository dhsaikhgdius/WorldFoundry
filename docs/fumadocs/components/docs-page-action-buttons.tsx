'use client';

import { Check, Copy } from 'lucide-react';
import { useState } from 'react';

export function DocsCopyMarkdownButton({
  label,
  copiedLabel,
  markdownUrl,
}: {
  label: string;
  copiedLabel: string;
  markdownUrl: string;
}) {
  const [copied, setCopied] = useState(false);

  async function onCopy() {
    try {
      const response = await fetch(markdownUrl);
      const text = await response.text();
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      window.open(markdownUrl, '_blank', 'noopener,noreferrer');
    }
  }

  return (
    <button type="button" className="pi-doc-action-chip" onClick={onCopy}>
      {copied ? <Check aria-hidden="true" size={14} strokeWidth={2} /> : <Copy aria-hidden="true" size={14} strokeWidth={2} />}
      <span>{copied ? copiedLabel : label}</span>
    </button>
  );
}
