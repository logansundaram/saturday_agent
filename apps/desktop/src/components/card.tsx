interface CardProps{
    header: string;
    body: string;
};


export default function Card({header, body} : CardProps){
    return (
        <div>
            <h1 className="accent">
                {header}
            </h1>
            <p>
                {body}
            </p>
        </div>
    )
}
