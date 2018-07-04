import json, os, dataset, errno, xml.etree.ElementTree
from Bio.Entrez import efetch

# Collect the ids of documents and retrieve their xml form in order to extract the abstract of each document
def scrap_PubMeds(documents):
    ids = []
    for document_list in documents:
        for document in document_list:
            ids.append(document.rsplit('/', 1)[-1])

    print("Number of total PubMeds to be extracted: ", len(ids))

    # Retrieve abstract from each document and create relevant external files
    doc_counter = 0
    for docid in ids:
        doc_counter += 1
        print(docid, " : ", doc_counter, "/", len(ids))
        try:
            os.makedirs("Pubmeds/Abstracts/" + docid)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # Create xml file for the current document
        xml_file = open("PubMeds/Abstracts/" + docid + "/" + docid + ".xml", "w")
        abstract_plain_text = get_XML(docid)
        if abstract_plain_text is not None:
            xml_file.write(abstract_plain_text)
        xml_file.close

        # Create text files with abstract and title
        get_Abstract(docid)
        get_Title(docid)

# Retrieve XML form for a specific document
def get_XML(doc_id):
    handle = efetch(db = 'pubmed', id = doc_id, retmode = 'text', rettype = 'xml')
    return handle.read()

# Extract document's abstract from XML form
def get_Abstract(docid):
    ft = open("Pubmeds/Abstracts/" + docid + "/" + docid + "_abstract.txt", 'w').close()
    try:
        PubmedArticleSet = xml.etree.ElementTree.parse("Pubmeds/Abstracts/" + docid + "/" + docid + ".xml").getroot()
        for PubmedArticle in PubmedArticleSet:
            for MedlineCitation in PubmedArticle:
                if MedlineCitation.tag == "MedlineCitation":
                    for Article in MedlineCitation:
                        for Abstract in Article:
                            if Abstract.tag == "Abstract":
                                for AbstractText in Abstract:
                                    if AbstractText.text is not None:
                                        file = open("PubMeds/Abstracts/" + docid + "/" + docid + "_abstract.txt", 'a')
                                        file.write(AbstractText.text)
                                        file.close()
    except:
        print("Error during XML parsing!!!")
        file = open("PubMeds/Abstracts/" + docid + "/" + docid + "_abstract.txt", 'a')
        file.write("")
        file.close()



# Extract document's title from XML form
def get_Title(docid):
    ft = open("Pubmeds/Abstracts/" + docid + "/" + docid + "_title.txt", 'w').close()
    try:
        PubmedArticleSet = xml.etree.ElementTree.parse("Pubmeds/Abstracts/" + docid + "/" + docid + ".xml").getroot()
        for PubmedArticle in PubmedArticleSet:
            for MedlineCitation in PubmedArticle:
                if MedlineCitation.tag == "MedlineCitation":
                    for Article in MedlineCitation:
                        if Article in MedlineCitation:
                            if Article.tag == "Article":
                                for ArticleTitle in Article:
                                    if ArticleTitle.tag == "ArticleTitle":
                                        file = open("PubMeds/Abstracts/" + docid + "/" + docid + "_title.txt", 'a')
                                        file.write(ArticleTitle.text)
                                        file.close()
    except:
        file = open("PubMeds/Abstracts/" + docid + "/" + docid + "_title.txt", 'a')
        file.write("")
        file.close()




# ***** Main Script body *****

questions, documents, beginSections, endSections, concepts, questions_ids, types = [], [], [], [], [], [], []

# Path of the BioASQ json file
fname = 'BioASQ_json_files/phaseB_dry-run_.json'
print("Accesing ", fname)
with open(fname) as data_file:
    data = json.load(data_file)

# (Warning) Comment whole block of code if you have already download and extract the abstracts
for question in data['questions']:
    questions.append(question.get('body'))
    documents.append(question.get('documents'))
print("#questions: ", len(questions))
scrap_PubMeds(documents)

# retrieve info for each question
for question in data['questions']:
    snippets = []
    #questions.append(question.get('body'))
    #documents.append(question.get('documents'))
    concepts.append(question.get('concepts'))
    questions_ids.append(question.get('id'))
    types.append(question.get('type'))
    snippets.append(question.get('snippets'))
    if None not in snippets:
        for element in snippets:
            for e in element:
                relevant_snippet = e.get('text')
                docid = e.get('document').rsplit('/', 1)[-1]

                # Create dataset
                if os.path.isfile("Pubmeds/Abstracts/" + docid + "/" + docid + "_abstract.txt"):
                    dataset.createTrainSetForm(question.get('id'), question.get('body'), docid, relevant_snippet)
                    #dataset.createTestSetForm(question.get('id'), question.get('body'), docid)












