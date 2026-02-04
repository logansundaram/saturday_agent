import Card from "../components/card"
import Monitor from "../components/monitor"
import LeftNav from "../components/left_nav"
import { useState } from "react"

export default function Dashboard(){
    const [page, setPage] = useState("chat");


    return (
        <div>
            <LeftNav/>
            {page === "chat" ? (
                <Card header="chat" body="this is a test card"/>
            ) : page === "models" ? (
                <Card header="model" body="this is a test card"/>
            ) : null}
            <Monitor/>
        </div>
    )
}
