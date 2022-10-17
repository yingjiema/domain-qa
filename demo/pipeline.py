from haystack.nodes import FARMReader
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline

# For test purpose
text1 = "Metabolic syndrome is a cluster of conditions that occur together, increasing your risk of heart disease, stroke and type 2 diabetes. These conditions include increased blood pressure, high blood sugar, excess body fat around the waist, and abnormal cholesterol or triglyceride levels. Having just one of these conditions doesn't mean you have metabolic syndrome. But it does mean you have a greater risk of serious disease. And if you develop more of these conditions, your risk of complications, such as type 2 diabetes and heart disease, rises even higher. Metabolic syndrome is increasingly common, and up to one-third of U.S. adults have it. If you have metabolic syndrome or any of its components, aggressive lifestyle changes can delay or even prevent the development of serious health problems."
text2 = "Rheumatoid arthritis is a chronic inflammatory disorder that can affect more than just your joints. In some people, the condition can damage a wide variety of body systems, including the skin, eyes, lungs, heart and blood vessels. An autoimmune disorder, rheumatoid arthritis occurs when your immune system mistakenly attacks your own body's tissues. Unlike the wear-and-tear damage of osteoarthritis, rheumatoid arthritis affects the lining of your joints, causing a painful swelling that can eventually result in bone erosion and joint deformity. The inflammation associated with rheumatoid arthritis is what can damage other parts of the body as well. While new types of medications have improved treatment options dramatically, severe rheumatoid arthritis can still cause physical disabilities.Rheumatoid arthritis is a chronic inflammatory disorder that can affect more than just your joints. In some people, the condition can damage a wide variety of body systems, including the skin, eyes, lungs, heart and blood vessels. An autoimmune disorder, rheumatoid arthritis occurs when your immune system mistakenly attacks your own body's tissues. Unlike the wear-and-tear damage of osteoarthritis, rheumatoid arthritis affects the lining of your joints, causing a painful swelling that can eventually result in bone erosion and joint deformity. The inflammation associated with rheumatoid arthritis is what can damage other parts of the body as well. While new types of medications have improved treatment options dramatically, severe rheumatoid arthritis can still cause physical disabilities."
text3 = "Thyroid cancer is cancer that develops from the tissues of the thyroid gland.[1] It is a disease in which cells grow abnormally and have the potential to spread to other parts of the body.[7][8] Symptoms can include swelling or a lump in the neck.[1] Cancer can also occur in the thyroid after spread from other locations, in which case it is not classified as thyroid cancer.[3]"
text4 = "Season it well, and season it early if you've got time. Prime rib has plenty of flavor on its own, so there's no real need to add much more than a good heavy sprinkling of salt and pepper. If you're able to plan ahead, it's best to season your prime rib with salt at least the day before, and up to four days ahead of roasting, letting it sit on a rack in your fridge uncovered. This will allow time for the salt to penetrate and season more deeply while also drying out the surface, which will lead to better browning during roasting."
DOCS = [{"content": text1}, {"content": text2}, {"content": text3}, {"content": text4 }]
MODEL = "dmis-lab/biobert-base-cased-v1.2"

class QAPipeline(object):
    def __init__(self):
        self.docs = DOCS
        self.model = MODEL

    def create_pipeline(self):
        document_store = InMemoryDocumentStore()
        document_store.write_documents(self.docs)

        retriever = TfidfRetriever(document_store=document_store)
        reader = FARMReader(model_name_or_path=self.model, use_gpu=True)
        pipe = ExtractiveQAPipeline(reader, retriever)
        self.pipeline = pipe

    def predict(self, query):
        prediction = self.pipeline.run(
            query=query, 
            params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
        )
        return prediction


