from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

# Neural Net

class Encoder_GRU(nn.Module):

	def __init__(self, hidden_size, embedding, dropout, n_layers = 2):

		super(Encoder, self).__init__()

		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.embedding = embedding
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
							dropout = dropout, bidirectional = True)

	def forward(self, input_seq, input_lengths, hidden = None):

		embedded = self.embedding(input_seq)
		packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
		outputs, hidden = self.gru(packed, hidden)
		outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
		outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

		return outputs, hidden

class Encoder_LSTM(nn.Module):

	def __init__(self, hidden_size, embedding, dropout, n_layers = 2):

		super(Encoder, self).__init__()

		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.embedding = embedding
		self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers,
							dropout = dropout, bidirectional = True)

	def forward(self, input_seq, input_lengths, hidden = None):

		embedded = self.embedding(input_seq)
		packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
		outputs, hidden = self.gru(packed, hidden)
		outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
		outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

		return outputs, hidden

# Luong Attention

class LuongAttn(nn.Module):

	def __init__(self, attn_method, hidden_size):

		super(LuongAttn, self).__init__()

		self.attn_method = attn_method

		if self.attn_method not in ['dot', 'general', 'concat']:
			raise ValueError(self.attn_method, 'is not an appropriate method')

		self.hidden_size = hidden_size

		if self.attn_method == 'general':
			self.attn = nn.Linear(self.hidden_size, hidden_size)
		elif self.attn_method == 'concat':
			self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
			self.v = nn.Parameter(torch.FloatTensor(hidden_size))

	def dot_score(self, hidden, encoder_output):
		return torch.sum(hidden * encoder_output, dim = 2)

	def general_score(self, hidden, encoder_output):
		energy = self.attn(encoder_output)
		return torch.sum(hidden * energy, dim = 2)

	def concat_score(self, hidden, encoder_output):
		energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), 
										encoder_output), 2)).tanh()
		return torch.sum(self.v * energy, dim = 2)

	def forward(self, hidden, encoder_outputs):
		if self.attn_method == 'general':
			attn_generies = self.general_score(hidden, encoder_outputs)
		elif self.attn_method == 'concat':
			attn_energies = self.concat_score(hidden, encoder_outputs)
		elif self.attn_method == 'dot':
			attn_energies = self.dot_score(hidden, encoder_outputs)

		attn_energies = attn_energies.t()

		return F.softmax(attn_energies, dim = 1).unsqueeze(1)

class Decoder_GRU(nn.Module):

	def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers = 2, dropout = 0):

		super(Decoder, self).__init__()

		self.attn_model = attn_model
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout

		self.embedding = embedding
		self.embedding_dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout = dropout)
		self.concat = nn.Linear(hidden_size * 2, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.attn = LuongAttn(attn_model, hidden_size)

	def forward(self, input_seq, last_hidden, encoder_outputs):

		embedded = self.embedding(input_seq)
		embedded = self.embedding_dropout(embedded)
		gru_output, hidden = self.gru(embedded, last_hidden)
		attn_weights = self.attn(gru_output, encoder_outputs)
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
		gru_output = gru_output.squeeze(0)
		context = context.squeeze(1)
		concat_input = torch.cat((gru_output, context), 1)
		concat_output = torch.tanh(self.concat(concat_input))
		output = self.out(concat_output)
		output = F.softmax(output, dim = 1)

		return output, hidden

class Decoder_LSTM(nn.Module):

	def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers = 2, dropout = 0):

		super(Decoder, self).__init__()

		self.attn_model = attn_model
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout

		self.embedding = embedding
		self.embedding_dropout = nn.Dropout(dropout)
		self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout = dropout)
		self.concat = nn.Linear(hidden_size * 2, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.attn = LuongAttn(attn_model, hidden_size)

	def forward(self, input_seq, last_hidden, encoder_outputs):

		embedded = self.embedding(input_seq)
		embedded = self.embedding_dropout(embedded)
		lstm_output, hidden = self.lstm(embedded, last_hidden)
		attn_weights = self.attn(lstm_output, encoder_outputs)
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
		lstm_output = lstm_output.squeeze(0)
		context = context.squeeze(1)
		concat_input = torch.cat((lstm_output, context), 1)
		concat_output = torch.tanh(self.concat(concat_input))
		output = self.out(concat_output)
		output = F.softmax(output, dim = 1)

		return output, hidden







