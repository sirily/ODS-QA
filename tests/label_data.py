import pandas as pd
from os import system, name 

def clear(): 
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
  
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 

if __name__ == "__main__":
    df = pd.read_parquet('data/qq_sim.parquet')
    cont_flag = 'y'
    while cont_flag == 'y':
        print('Вам будут даны пять вопросов. К каждому из вопросов даётся список из 10 других вопросов. Отметьте, какие из этих вопросов похожи на изначальный')
        sample = df[df['checked'] == False].sample(n = 5)
        true_sim_ids = []
        for i, row in sample.iterrows():
            cand = row['candidates'].tolist()
            scores = []
            for i, q in enumerate(cand):
                clear()
                print(f"Вопрос номер {i+1}:\n{row['question']}")
                print(f'{i}\t{q}')
                print('-'*150) 
                score = int(input('Оцените похожесть этого вопроса на вопрос выше от 0 до 10: '))
                scores.append(score)

            for i, score in enumerate(scores):
                chosen = []
                chosen_scores = []
                if score >= 5:
                    chosen.append(cand[i])
                    chosen_scores.append(score)

            #add chosen questions to dataframe
            ids_str = ''
            scores_str = ''
            for i, q in enumerate(chosen):
                if len(ids_str) > 0:
                    ids_str += ', '
                    scores_str += ', '
                ids_str += str(df['question_ids_in_clean_df'][df['question'] == q].tolist()).strip("['']")
                scores_str += str(chosen_scores[i])
            df['similiar_questions_ids_in_clean_df'][df['question'] == row['question']] = ids_str
            df['scores'][df['question'] == row['question']] = scores_str

            #add checked flag
            df['checked'][df['question'] == row['question']] = True

        cont_flag = input('Продолжить? [y/n]')

    #save df
    df.to_parquet('data/qq_sim.parquet', compression='brotli', index=False)
                 