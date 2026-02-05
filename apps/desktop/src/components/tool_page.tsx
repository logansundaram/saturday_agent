import ModelCard from "./model_card";

export default function ToolPage(){
    return (
        <div className="bg-panel px-50 h-screen">
            <h1 className="text-left p-4 text-9xl">
                Tools
            </h1>

            <div className="text-left p-4">

                <div className="grid grid-cols-3 w-full place-items-center">
                    <ModelCard/>
                    <ModelCard/>
                    <ModelCard/>
                </div>

            </div>

        </div>
    )
}