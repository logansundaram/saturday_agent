interface Monitor_PillProps{
    header : string;
    name : string;
    useage: string;
}


//cut useage line and instead use the bar to convey useage info, red maxed out, orange over 90 percent, green under 80 percent
export default function Monitor_Pill({header, name, useage} : Monitor_PillProps){
    return (
        <div className="border-1 m-2 p-4 grid grid-rows-3 gap-2 rounded-lg text-left">
            <div className="">
                {header} : {name}
            </div>
            <div>
                Useage : {useage}%
            </div>
            <div className="mt-2 h-2 w-full rounded bg-slate-200">
                <span className="block h-full rounded bg-green-500"/>
            </div>
        </div>
    )
}
