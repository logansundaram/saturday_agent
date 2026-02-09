import * as React from "react";

type TextareaProps = React.TextareaHTMLAttributes<HTMLTextAreaElement>;

const cx = (...classes: Array<string | undefined | false>) =>
  classes.filter(Boolean).join(" ");

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        ref={ref}
        className={cx(
          "flex w-full rounded-md border border-subtle bg-transparent px-3 py-2 text-sm text-primary shadow-sm transition placeholder:text-secondary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#6d28d9]/40",
          className
        )}
        {...props}
      />
    );
  }
);
Textarea.displayName = "Textarea";

export { Textarea };
