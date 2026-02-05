import Monitor_Pill from "./monitor_pill"
import { useState } from "react"

export default function Monitor(){
    const [open, setOpen] = useState(true);

    return (
        <div className="fixed top-0 right-0 h-full bg-root w-50">
            <div className="p-4 flex justify-between">
                <div>
                    System Monitoring
                </div>
                <span className="text-right" onClick={() => {
                    setOpen(!open);
                }}>
                    {open ? <>-</> : <>+</>}
                </span> 
            </div>

            
            {open ? (
                <>
                    <Monitor_Pill header="GPU" usage={85} name="Nvidia 5090"/>
                    <Monitor_Pill header="CPU" usage={32} name="AMD 9950X"/>
                    <Monitor_Pill header="RAM" usage={45} name="32GB"/>
                    <Monitor_Pill header="VRM" usage={100} name="24GB"/>

                </>
            ) : null}
           

        </div>
    )
}
