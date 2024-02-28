import { WriteMode, connect } from "vectordb";
import { configDotenv } from "dotenv";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
configDotenv()

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 300,
    chunkOverlap: 100,
    separators: ["\n"],
});

const getDb = async () => {
    try {
        const db = await connect("./vectordb-dev/db");
        const data = await fetch('https://wpengine.com/wp-json/wp/v2/posts/149490?_fields=id,date,guid,modified,slug,status,type,link,title,content,excerpt,comment_status,template,format,yoast_head_json')
        const response = await data.json()
        console.log(typeof response)
        const model = new HuggingFaceTransformersEmbeddings({
            modelName: "Xenova/all-MiniLM-L6-v2",
        });

        let objectData = []
        for (let key in response) {
            let object = JSON.stringify(response[key])
            const cleanedJson = object.replace(/[\{\}\"]/g, '').replace(/:/g, ' : ').replace(/,/g, '\n');
            const splitArray = await splitter.splitText(cleanedJson)
            for (let chunck of splitArray) {
                const embedded = await model.embedQuery(chunck)
                objectData.push({ vector: embedded, data: chunck })
            }
        }
        // const table = await db.openTable("testTable2");
        const table = await db.createTable("testTable", objectData, { writeMode: WriteMode.Overwrite })
        const query = "is it public"
        const vectorQuery = await model.embedQuery(query)
        const result = await table.search(vectorQuery).limit(3).execute()
        for (let i of result) {
            console.log(i['data'] + "\n")
        }

    } catch (e) {
        console.log(e)
    }
}

getDb()