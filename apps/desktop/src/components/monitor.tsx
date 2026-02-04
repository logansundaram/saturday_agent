import Monitor_Pill from "./monitor_pill"
import { useState } from "react"

export default function Monitor(){
    const [open, setOpen] = useState(true);

    return (
        <div className="fixed top-0 right-0 h-full bg-slate-200 w-50">
            <div className="p-4 flex justify-between">
                <div>
                    System Monitoring
                </div>
                <span className="text-right" onClick={() => {
                    setOpen(!open);
                }}>
                    {open ? <>+</> : <>-</>}
                </span> 
            </div>

            
            {open ? (
                <>
                    <Monitor_Pill header="GPU" useage="85" name="Nvidia 5090"/>
                    <Monitor_Pill header="CPU" useage="32" name="AMD 9950X"/>
                    <Monitor_Pill header="RAM" useage="45" name="32GB"/>
                </>
            ) : null}
           

        </div>
    )
}
