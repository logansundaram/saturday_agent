import ModelCard from "./model_card";

export default function ModelPage(){
    return (
        <div className="bg-panel section-base">
            <div className="section-hero">
                <h1 className="section-header">
                    Models
                </h1>
                <p className="section-framer">
                    Local-first models by default, with optional cloud APIs. Choose the right model for each task, on your terms.
                </p>
            </div>


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