import * as React from "react";

type BadgeProps = React.HTMLAttributes<HTMLDivElement>;

const cx = (...classes: Array<string | undefined | false>) =>
  classes.filter(Boolean).join(" ");

function Badge({ className, ...props }: BadgeProps) {
  return (
    <div
      className={cx(
        "inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium",
        className
      )}
      {...props}
    />
  );
}

export { Badge };
