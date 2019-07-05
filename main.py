from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from sheldonbot import models
from sheldonbot import loss
from sheldonbot import batch_gen
from sheldonbot import preprocessing
from sheldonbot import vocabulary

import torch.nn as nn
import torch
import os



def train(input_variable, lengths, target_variable, mask, max_target_len,
          encoder, decoder, embedding, encoder_optimizer, decoder_optimizer,
          batch_size, clip, max_length = MAX_LENGTH, sos_token, mask_loss):
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    loss = 0
    print_losses = []
    n_totals = 0
    
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
    
    decoder_input = torch.LongTensor([[sos_token for _ in range(batch_size)]])
    
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, 
                                                     decoder_hidden, 
                                                     encoder_outputs)
            decoder_input = target_variable[t].view(1, -1)
            mask_loss, nTotal = mask_loss(decoder_output, target_variable[t],
                                            mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item()*nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, 
                                                     decoder_hidden, 
                                                     encoder_outputs)
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item()*nTotal)
            n_totals += nTotal
            
    loss.backward()
    
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return sum(print_losses) / n_totals

def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer,
               decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, 
               save_dir, n_iteration, batch_size, print_every, save_every, clip,
               corpus_name):
    
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)]) 
                        for _ in range(n_iteration)]
    
    
    print('Initializing...')
    start_iteration = 1
    print_loss = 0
    #if loadFilename:
    #    start_iteration = checkpoint['iteration'] + 1
        
    print('Training...')
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch
        loss = train(input_variable, lengths, target_variable, mask, 
                     max_target_len, encoder, decoder, embedding, 
                     encoder_optimizer, decoder_optimizer, batch_size, clip)
        
        print_loss += loss
        
        if iteration % print_every == 0:
            print_loss_avg = print_loss/print_every
            print('Iteration: {}; Percent complete: {:.1f}%; Average loss:{:.4f}'.format(iteration, 
                  iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0
            
            
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, 
                                     '{}_{}_{}'.format(encoder_n_layers, 
                                      decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            torch.save({'iteration': iteration,
                        'en': encoder.state_dict(),
                        'de': decoder.state_dict(),
                        'en_opt': encoder_optimizer.state_dict(),
                        'de_opt': decoder_optimizer.state_dict(),
                        'loss': loss,
                        'voc_dict': voc.__dict__,
                        'embedding': embedding.state_dict()}, 
                os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


# Prepping for training
corpus_name = 'movies_bigbang'
movies_datafile = os.path.join('data', 'movie_lines.txt')
big_bang_datafile =  os.path.join('data', 'BigBangTheory_pairs.txt')
voc, movie_pairs, bigbang_pairs = preprocessing.loadPrepareData(corpus_name, movies_datafile, big_bang_datafile) 

model_name = 'movies_bigbang_model'
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
checkpoint_iter = 10000
movies_train_savdir = os.path.join('movies_trained_models', '2_layer_gru')
bigbang_train_savdir = os.path.join('bigbang_trained_models', '2_layer_gru')
voc_dir = os.path.join('data', 'movies_bigbang.pkl')


with open(voc_dir, 'wb') as output:
    pickle.dump(voc, output, pickle.HIGHEST_PROTOCOL)


# Starting the training
embedding = nn.Embedding(voc.num_words, hidden_size)





















