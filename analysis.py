import json
import re

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def clean_options(options):
    cleaned_options = {}
    for option in options:
        try:
            letters = re.findall('[A-F]', option['extract_result'])
        except:
            letters = re.findall('[A-F]', option['model_answer'])

        letters = sorted(list(set(letters)))
        cleaned_options[option['id']] = ''.join([letter.upper() for letter in letters])

    return cleaned_options

def calculate_accuracy(answers, options, flag=True):
    correct_count = 0
    total_questions = len(options)
    total_questions = 0

    static_dict = {}
    uncorrected_1 = []
    uncorrected_2 = []

    for question in answers:
        question_id = question['id']
        cleaned_answer = ''.join(filter(str.isalpha, question['answer']))
        cleaned_option = options.get(question_id, '').upper()

        if len(cleaned_answer) == 1 and len(cleaned_option)>1:
            cleaned_option = cleaned_option[0]

        if cleaned_option == '':
            pass
        elif cleaned_answer == cleaned_option:
            correct_count += 1
            total_questions += 1
            question_type = question['exam_type']
            if question_type in static_dict.keys():
                static_dict[question_type] = static_dict[question_type] + 1
            else:
                static_dict[question_type] = 1
                
        else:
            total_questions += 1
            
            if flag:
                uncorrected_1.append(question_id)
            else:
                uncorrected_2.append(question_id)

    accuracy = (correct_count / total_questions) * 100
    # print(str(correct_count) + '/' + str(total_questions))
    return accuracy

def evaluate(options_file_path):

    options = load_json(options_file_path)

    cleaned_options = clean_options(options)

    accuracy = calculate_accuracy(options, cleaned_options, False)

    return round(accuracy, 2)

if __name__ == "__main__":
    from utils import get_args
    args = get_args()
    if args.dataset_name != 'Clin':
        accuracy = evaluate(args.ex_result_path)
        print(f"Accurary: {accuracy}%")
    else:
        from clinical import evaluate_clinical
        evaluate_clinical(args)