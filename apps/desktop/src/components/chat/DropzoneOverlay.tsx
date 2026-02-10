type DropzoneOverlayProps = {
  visible: boolean;
};

export default function DropzoneOverlay({ visible }: DropzoneOverlayProps) {
  if (!visible) {
    return null;
  }

  return (
    <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center rounded-2xl border-2 border-dashed border-gold/70 bg-black/45">
      <p className="rounded-full border border-gold/40 bg-black/60 px-4 py-2 text-sm text-[#f2d588]">
        Drop image files to attach
      </p>
    </div>
  );
}
