import { Button } from "../ui/button";

export type ChatAttachment = {
  artifactId: string;
  name: string;
  mime: string;
  size: number;
  sha256: string;
  previewUrl: string;
};

type AttachmentsBarProps = {
  attachments: ChatAttachment[];
  onRemove: (artifactId: string) => void;
};

export default function AttachmentsBar({
  attachments,
  onRemove,
}: AttachmentsBarProps) {
  if (attachments.length === 0) {
    return null;
  }

  return (
    <div className="mb-3 flex flex-wrap gap-2">
      {attachments.map((attachment) => (
        <div
          key={attachment.artifactId}
          className="flex items-start gap-2 rounded-xl border border-subtle bg-black/20 p-2"
        >
          <img
            src={attachment.previewUrl}
            alt={attachment.name || "Attachment preview"}
            className="h-14 w-14 rounded-md border border-subtle object-cover"
          />
          <div className="min-w-0">
            <p className="max-w-[10rem] truncate text-xs text-primary">{attachment.name}</p>
            <p className="text-[11px] text-secondary">
              {(attachment.size / 1024).toFixed(1)} KB
            </p>
            <Button
              type="button"
              className="mt-1 h-6 rounded-full border border-subtle bg-transparent px-2 text-[11px] text-secondary hover:text-primary"
              title="Remove attachment"
              onClick={() => onRemove(attachment.artifactId)}
            >
              X
            </Button>
          </div>
        </div>
      ))}
    </div>
  );
}
