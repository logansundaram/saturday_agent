import Chattile from "./chat_tile";
import { useState } from "react";

export default function Chats(){
    const [open, setOpen] = useState(false);
    return(
        <div className="text-left p-2">
            <div className="flex justify-between">
                <p>
                    Chats
                </p>
                <span className="text-right" onClick={() => {
                    setOpen(!open);
                }}>
                    {open ? <>-</> : <>+</>}
                </span> 
            </div>

            { open ?
                <>
                    <Chattile/>
                    <Chattile/>
                    <Chattile/>
                    <Chattile/>
                </>
                :
                <></>
            }
        </div>
    )
}