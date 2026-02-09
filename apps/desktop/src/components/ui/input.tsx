import * as React from "react";

type InputProps = React.InputHTMLAttributes<HTMLInputElement>;

const cx = (...classes: Array<string | undefined | false>) =>
  classes.filter(Boolean).join(" ");

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type = "text", ...props }, ref) => {
    return (
      <input
        ref={ref}
        type={type}
        className={cx(
          "flex h-10 w-full rounded-md border border-subtle bg-transparent px-3 py-2 text-sm text-primary shadow-sm transition placeholder:text-secondary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#6d28d9]/40",
          className
        )}
        {...props}
      />
    );
  }
);
Input.displayName = "Input";

export { Input };
