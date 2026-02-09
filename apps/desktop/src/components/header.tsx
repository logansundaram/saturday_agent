interface HeaderProps {
    hero: string;
    framer: string;
}

export default function Header({hero, framer}: HeaderProps) {
    return (
        <div className="header">
            <div className="section-hero">
                <h1 className="section-header">
                    {hero}
                </h1>
                <p className="section-framer">
                    {framer}
                </p>
            </div>
        </div>        
    ) 
}