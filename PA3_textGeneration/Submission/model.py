from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset
import torch
from tqdm import tqdm
import numpy as np

device = "cuda"

def get_transformer_model():

	# Feel free to change models if having memory issue
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
  
  #To remove further issues due to eos_token
	model.config.pad_token_id = model.config.eos_token_id

	# 'pt' for PyTorch, 'tf' for TensorFlow
	framework = 'pt'

	return TransformerModel(model, tokenizer, framework)


class TransformerModel(object):

	def __init__(self, model, tokenizer, framework='pt'):

		self.model = model
		self.tokenizer = tokenizer
		self.framework = framework

		##### Feel free to add more attributes here if needed #####


	def generate_text(self, prompt, max_new_tokens=10, num_return_sequences=1):
		"""
		The method generates the complementary text for a given starting
		text, i.e., the prompt.

		Args:
			prompt: the starting text as a string
			max_length [optional]: the max length of the generated text

		Return:
			results: the generated text as a string.
		"""

		##### Your code here #####
		if max_new_tokens == 2:
      # Will be used for Part-2 contrastive search, As the reviews provided are not so straight forward it would be better to go with this technique
			encoded_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
			generated_output = self.model.generate(encoded_ids,  penalty_alpha=0.35, top_k=20, max_new_tokens=max_new_tokens,num_return_sequences=num_return_sequences)
			output_string = self.tokenizer.decode(generated_output[0], skip_special_tokens=True)

			return output_string
		
		input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
		
		print("\n Task1: Generation")
    # Simple text generator 
		naive_output = self.model.generate(input_ids, do_sample = False, max_new_tokens=max_new_tokens,num_return_sequences=num_return_sequences)
		output_string = self.tokenizer.decode(naive_output[0], skip_special_tokens=True)
		#print(output_string)
		results=[]
		results= [output_string]

		print("\n Greedy Sampling:")
    # Trying out several techniques, Greedy sampling is done in the following
		output = self.model.generate(input_ids, do_sample = False, max_new_tokens=max_new_tokens,num_return_sequences=num_return_sequences)
		output_string = self.tokenizer.decode(output[0], skip_special_tokens=True)
		print(output_string)
		results.append(output_string)

		print("\n Multinomial Sampling:")
    # The following code is for multinomial sampling 
		output = self.model.generate(input_ids, max_new_tokens=max_new_tokens, num_beams=1, num_return_sequences=num_return_sequences, do_sample=True)
		output_string = self.tokenizer.decode(output[0], skip_special_tokens=True)
		print(output_string)
		results.append(output_string)

		print("\n Beam-search Multinomial Sampling:")
    # the following is the code for beam-search multinomial sampling 
		output = self.model.generate(input_ids, max_new_tokens=max_new_tokens, num_beams=10, num_return_sequences=num_return_sequences, do_sample=True)
		output_string = self.tokenizer.decode(output[0], skip_special_tokens=True)
		print(output_string)
		results.append(output_string)

		print("\n Beam-search Sampling:")
    # the following is the code for beam-search sampling
		output = self.model.generate(input_ids, num_beams=10, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences, do_sample=False)
		output_string = self.tokenizer.decode(output[0], skip_special_tokens=True)
		print(output_string)
		results.append(output_string)

		print("\n Diverse Beam-search Sampling:")
    # the following is the code for Diverse Beam-search sampling
		output = self.model.generate(input_ids, num_beams=10, max_new_tokens=max_new_tokens, num_beam_groups=5)
		output_string = self.tokenizer.decode(output[0], skip_special_tokens=True)
		print(output_string)
		results.append(output_string)

		print("\n Diverse Beam-search Sampling with temperature:")
    # the following is the code for Diverse beam-search sampling with temperature
		output = self.model.generate(input_ids, num_beams=10, max_new_tokens=max_new_tokens, num_beam_groups=5, temperature=1.4)
		output_string = self.tokenizer.decode(output[0], skip_special_tokens=True)
		print(output_string)
		results.append(output_string)

		print("\n")
		##### Code done #####
		results = "\n".join(results)

		return results


	def evaluate_ppl(self, dataset):
		"""
		The method for evaluating the perplexity score on given datasets,
		e.g., WikiText-2.

		Args:
			dataset: a `datasets.Dataset' instance from Huggingface

		Return:
			score: A float number. The perplexity score.
		"""

		##### Your code here #####
		encodings  = self.tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
		max_length = self.model.config.n_positions
		stride     = 64  #128, 256, 512 
		seq_len = encodings.input_ids.size(1)

		nlls = []
		prev_end_loc = 0
		device = "gpu"
		model_id = "gpt2"
		if torch.cuda.is_available():
			device = "cuda:0"
		else:
			device = "cpu"
		device = torch.device(device)
		model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
		tokenizer = GPT2Tokenizer.from_pretrained(model_id)
		for begin_loc in tqdm(range(0, seq_len, stride)):
				end_loc = min(begin_loc + max_length, seq_len)
				trg_len = end_loc - prev_end_loc  
				input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
				target_ids = input_ids.clone()
				target_ids[:, :-trg_len] = -100

				with torch.no_grad():
						outputs = model(input_ids, labels=target_ids)
						neg_log_likelihood = outputs.loss * trg_len

				nlls.append(neg_log_likelihood)

				prev_end_loc = end_loc
				if end_loc == seq_len:
						break

		ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
		

		##### Code done #####

		return ppl


	def get_template(self, doc, lbl):
		##### Write your own template below #####
		template = 'Review: \"%s\"\nSentiment: %s' %(doc, lbl)
		##### Template done #####

		return template


	def fewshot_sentiment(self, trainSet, test_doc):
		"""
		Taking advantage of the language model to perform sentiment analysis.

		Args:
			trainSet: List of tuples. Each tuple is a pair of (document, label),
					  where `document` is a string of the entire document and 
					  label is either 'positive' or 'negative'
			test_doc: String. The test document.
		Return:
			prediction: String. The predicted sentiment, 'positive' or 
						'negative'.
		"""

		prompt = ''
		for (doc, lbl) in trainSet:
			prompt += self.get_template(doc, lbl)
			prompt += '\n###\n'

		prompt += self.get_template(test_doc, "")

		# 'positive'/'negative' plus an EoS token
		prediction = self.generate_text(prompt, max_new_tokens=2)

		return prediction.split('\n###\n')[-1]


	def visualize_attention(self, trainSet, test_doc, layer=-1):
		"""
		(Bonus) Visualize how attention works in the fewshot sentiment analysis.

		Args:
			trainSet: List of tuples. Each tuple is a pair of (document, label),
					  where `document` is a string of the entire document and 
					  label is either 'positive' or 'negative'
			test_doc: String. The test document.
			layer: Integer. To speficify which attention layer to be visualized.
		Return:
			template: The template input to the language model.
			weights: 1D-Array. The attention score of each token in the template.
					 Values should be in [0,1], normalize if needed.
		"""

		prompt = ''
		for (doc, lbl) in trainSet:
			prompt += self.get_template(doc, lbl)
			prompt += '\n###\n'

		prompt += self.get_template(test_doc, "")

		##### Your code here #####

		weights = np.random.rand(len(prompt.split()))

		##### Code done #####
		assert len(prompt.split())==len(weights)

		return prompt, weights


	def finetune(self, trainSet):
		"""
		Taking advantage of the language model to perform sentiment analysis.

		Args:
			trainSet: List of tuples. Each tuple is a pair of (document, label),
					  where `document` is a string of the entire document and 
					  label is either 'positive' or 'negative'
		"""
		templates = [{"text": self.get_template(doc, lbl)} for doc, lbl in trainSet]
		dataset = Dataset.from_list(templates)
		# Use "left" truncation so that the sentiment is not truncated.
		map_tokenize = lambda x: self.tokenizer(x['text'], truncation_side='left')
		dataset = dataset.map(map_tokenize, batched=True)
		dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.1)

		##### Your code here #####



		##### Code done #####










