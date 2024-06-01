from imports import *

simple_prompt = """
  Task: Generate {n} questions and their corresponding optimal answers that could be considered as ground Truth from the given context.
  Just return the question as it is relevant to the context.

  Requirements:
    Create {n} comprehensive yet brief questions.
    The Questions must be unique and should not Overlap with the Available Questions.
    Ensure the questions has a difficulty level ranging from simple to moderate.
    Make sure each Question could be answered Completely from the Given Text Only.
    Provide the best possible answers that can be considered the ground truth for the generated questions.
    Make sure the Quality of the Answers must be Top Notch and Add as much Detial as Possible to make the Answer Engauging and Enhanced Quality.
    Attach the appropriate Metadata as Dictionary from the Provided COntext Documents.


  Response Format:
    Donot add any prefix or postfix respond the results in given blocks ``` ```
    ```
    [
    {{
      "question": "Generated Question",
      "ground_truth": "ground Truth",
      "contexts": ["context 01 text","context 02 text"],
      "metadata":[{{}},{{}}]
    }}
    ,
    {{
      "question": "Generated Question",
      "ground_truth": "ground Truth",
      "contexts": ["context 01 text","context 02 text"],
      "metadata":[{{}},{{}}]

    }}

    ]

    ```

  <available-questions>
    {available_questions}
  </available-questions>


  <context>
    {input}
  </context>

"""

reasoning_prompt = """
  Task: Generate {n} questions in terms of Qualitative/Quantitative/Analytical Reasoning Aspect and their corresponding optimal answers that could be considered as ground Truth from the given context documents.
  Just return the question as it is relevant to the context.

  Requirements:
    Create {n} Questions that could be used to Evaluate the Reasoning Capabilities.
    The Questions must be unique and should not Overlap with the Available Questions.
    Ensure the question should be a Analytical Reasoning Question from the Given Text.
    Make sure each Question could be answered Completely from the Given Text Only.
    Provide the best possible answers that can be considered the ground truth for the generated questions.
    Make sure the Quality of the Answers must be Top Notch and Add as much Detial as Possible to make the Answer Engauging and Enhanced Quality.
    The Context list Should be List of Exact Same Text Chunks from the given context.
    Attach the appropriate Metadata as Dictionary from the Provided COntext Documents.


  Response Format:
    Donot add any prefix or postfix respond the results in given blocks ``` ```
    ```
    [
    {{
      "question": "Generated Question",
      "ground_truth": "ground Truth",
      "contexts": ["context 01 text","context 02 text"],
      "metadata":[{{}},{{}}]
    }}
    ,
    {{
      "question": "Generated Question",
      "ground_truth": "ground Truth",
      "contexts": ["context 01 text","context 02 text"],
      "metadata":[{{}},{{}}]

    }}

    ]

    ```

  <available-questions>
    {available_questions}
  </available-questions>


  <context>
    {input}
  </context>


"""

multicontext_prompt = """
  Task: Generate {n} multicontext questions (may involves conditioning from Different Text Topics) and their corresponding optimal answers that could be considered as ground Truth from the given Context.
  Just return the question as it is relevant to the context.

  Requirements:
    Create {n} Questions that could be used to Evalaute the Multi Context answering Capabilities.
    The Questions must be unique and should not Overlap with the Available Questions.
    Ensure the questions should contain the Multiple/Merged Context References from the Given Text.
    Make sure each Question could be answered Completely from the Given Text Only.
    Provide the best possible answers that can be considered the ground truth for the generated questions.
    Make sure the Quality of the Answers must be Top Notch and Add as much Detial as Possible to make the Answer Engauging and Enhanced Quality.
    Attach the appropriate Metadata as Dictionary from the Provided COntext Documents.

  Response Format:
    Donot add any prefix or postfix respond the results in given blocks ``` ```
    ```
    [
    {{
      "question": "Generated Question",
      "ground_truth": "ground Truth",
      "contexts": ["context 01 text","context 02 text"],
      "metadata":[{{}},{{}}]
    }}
    ,
    {{
      "question": "Generated Question",
      "ground_truth": "ground Truth",
      "contexts": ["context 01 text","context 02 text"],
      "metadata":[{{}},{{}}]

    }}

    ]

    ```
  <available-questions>
    {available_questions}
  </available-questions>


  <context>
    {input}
  </context>

"""

testcases_prompts = {
    'simple': simple_prompt,
    'reasoning': reasoning_prompt,
    'multi_context': multicontext_prompt
}


def promptExecute(prompt, input_data, available_questions, n, evolution_type):
  prompt_template = prompt
  # Format the prompt with the given inputs
  prompt = prompt_template.format(input=input_data,
                                  available_questions=available_questions,
                                  n=n)

  # Call the OpenAI API to get the completion
  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[{
          "role": "system",
          "content": "You are a helpful assistant."
      }, {
          "role": "user",
          "content": prompt
      }],
      temperature=0.0  # Low temperature for more deterministic output
  )
  print(
      str(response.choices[0].message.content).replace("```",
                                                       "").replace("```", ""))
  cleaned_json_data = json.loads(
      str(response.choices[0].message.content).replace("```",
                                                       "").replace("```", ""))
  questions = []
  for entry in cleaned_json_data:
    questions.append(entry['question'])
    entry["evolution_type"] = evolution_type
  return cleaned_json_data, questions


def round_cases(total_cases):
  DEFAULT_DISTRIBUTION = {
      'simple': 0.5,
      'reasoning': 0.25,
      'multi_context': 0.25
  }

  # Calculate exact number of test cases for each type
  exact_cases = {
      key: value * total_cases
      for key, value in DEFAULT_DISTRIBUTION.items()
  }

  # Round to the nearest whole number
  rounded_cases = {
      key: math.floor(value)
      for key, value in exact_cases.items()
  }

  # Adjust the total to ensure it sums up to total_cases
  while sum(rounded_cases.values()) < total_cases:
    for key in sorted(rounded_cases,
                      key=lambda k: exact_cases[k] - rounded_cases[k],
                      reverse=True):
      if sum(rounded_cases.values()) < total_cases:
        rounded_cases[key] += 1
      else:
        break
  return rounded_cases


def makedocsstr(documents):
  docs_str = ""
  for i, item in enumerate(documents):
    docs_str += f"<Doc metadata={item.metadata}>\n" + str(
        item.page_content) + "\n</doc>\n"
  return docs_str


def Generate_Test_Cases(documents, total_cases):
  distributions = round_cases(total_cases)
  available_questions = []
  resultant_data = []
  for key, value in distributions.items():
    while True:
      try:
        if value > len(documents):
          documents_strs = makedocsstr(documents)
        else:
          documents_strs = makedocsstr(random.sample(documents, value))
        response, genearted_questions = promptExecute(
            testcases_prompts[key], documents_strs,
            '\n'.join(available_questions), value, key)
        break
      except:
        continue

    available_questions.extend(genearted_questions)
    resultant_data.extend(response)

  return pd.DataFrame(resultant_data)
