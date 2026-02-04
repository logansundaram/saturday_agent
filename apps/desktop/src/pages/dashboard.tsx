import Card from "../components/card"
import Monitor from "../components/monitor"
import LeftNav from "../components/left_nav"


export default function Dashboard(){
    return (
        <div>
            <LeftNav/>
            <Monitor/>
            <Card header="test" body="this is a test card"/>
        </div>
    )
}