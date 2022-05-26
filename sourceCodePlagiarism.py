from select import select
import streamlit as st
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def main():
    st.title("Source Code Plagiarism Detection")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    doc1 = col1.text_input("Give the Source Code File Extension:")
    doc2 = col2.text_input("Give the Plagiarism Detection Code File Extension:")
    try:
        with open(doc1) as f:
            doc1 = f.read().lower()
            doc1 = re.sub(r"\[.*\]|\{.*\}", "", doc1)
            doc1 = re.sub(r'[^\w\s]', "", doc1)
            doc1 = ''.join(filter(lambda x: not x.isdigit(), doc1))
            fileAfterCleaning1 = col3.text_input("Give the extension for the cleaned source data:")
        with open(doc2) as f:
            doc2 = f.read().lower()
            doc2 = re.sub(r"\[.*\]|\{.*\}", "", doc2)
            doc2 = re.sub(r'[^\w\s]', "", doc2)
            doc2 = ''.join(filter(lambda x: not x.isdigit(), doc2))
            fileAfterCleaning2 = col4.text_input("Give the extension for the cleaned detect data:")
            col5,col6,col7,col8,col9,col10,col11 = st.columns(7)

            if col8.button("SAVE"):
              y = [doc2]
              with open(fileAfterCleaning2,'w') as f:
                for line in y:
                    f.write(line)
                    f.write('\n')
              x = [doc1]
              with open(fileAfterCleaning1,'w') as f:
                for line in x:
                    f.write(line)
                    f.write('\n')
                    st.success("All done!!")
    except:
        pass 
    
    try:    
        documents = [fileAfterCleaning1,fileAfterCleaning2]
        documentTexts = [open(document).read() for document in  documents]

        vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
        similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])

        vectors = vectorize(documentTexts)
        vectorsList = list(zip(documents, vectors))
        
        def checkPlagiarism(vectorsList):
            for documentA, textVectorA in vectorsList:
                newVectors = vectorsList.copy()
                currentIndex = newVectors.index((documentA, textVectorA))
                del newVectors[currentIndex]
                for documentB , textVectorB in newVectors:
                    simScore = similarity(textVectorA, textVectorB)[0][1]
                    return simScore

        codeSimilarityScore = checkPlagiarism(vectorsList)

        similarityScore = ctrl.Antecedent(np.arange(0, 1, 0.1), 'similarity')
        score = ctrl.Consequent(np.arange(0, 1, 0.1), 'Score')
        similarityScore.automf(5)

        score['benzerlik_yok'] = fuzz.trimf(score.universe, [0, 0, 0.4])
        score['az_benzer'] = fuzz.trimf(score.universe, [0.4, 0.4, 0.55])
        score['orta_benzer'] = fuzz.trimf(score.universe, [0.55, 0.55, 0.7])
        score['benzer'] = fuzz.trimf(score.universe, [0.7, 0.7, 0.85])
        score['cok_benzer'] = fuzz.trimf(score.universe, [0.85, 0.85, 1])

        st.set_option('deprecation.showPyplotGlobalUse', False)

        rule1 = ctrl.Rule(similarityScore["poor"],score["benzerlik_yok"])
        rule2 = ctrl.Rule(similarityScore["mediocre"],score["az_benzer"])
        rule3 = ctrl.Rule(similarityScore["average"],score["orta_benzer"])
        rule4 = ctrl.Rule(similarityScore["decent"],score["benzer"])
        rule5 = ctrl.Rule(similarityScore["good"],score["cok_benzer"])

        scoringCtrl = ctrl.ControlSystem([rule1, rule2, rule3,rule4,rule5])
        scoring = ctrl.ControlSystemSimulation(scoringCtrl)
        scoring.input['similarity'] = codeSimilarityScore
        scoring.compute()

        option = st.selectbox(
        'Outputs',
        ['Plagiarism Ratio','Visualization Ratio in Output'])

        if 'Plagiarism Ratio' in option:
            st.write("Plagiarism Ratio: " +str(scoring.output['Score']))
        
        if 'Visualization Ratio in Output' in option:
            st.pyplot(score.view(sim=scoring))

    except:
        pass

if __name__ == '__main__':
	main()




		