""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """
import torch
import logging
import os
import copy
import json
from .utils_ner import DataProcessor,get_entities
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, labels, query_tokens, query_labels):
        self.guid = guid
        self.text_a = text_a
        self.labels = labels
        self.query_tokens = query_tokens
        self.query_labels = query_labels
    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids, start_ids,end_ids, subjects):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_ids = start_ids
        self.input_len = input_len
        self.end_ids = end_ids
        self.subjects = subjects

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_input_mask, all_segment_ids, all_start_ids,all_end_ids,all_lens = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    all_segment_ids = all_segment_ids[:, :max_len]
    all_start_ids = all_start_ids[:,:max_len]
    all_end_ids = all_end_ids[:, :max_len]
    return all_input_ids, all_input_mask, all_segment_ids, all_start_ids,all_end_ids,all_lens

def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer,
                                 cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=0,
                                 sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                 sequence_a_segment_id=0,sequence_b_segment_id=1,mask_padding_with_zero=True,do_lower_case=False,):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label2id = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        # print(example.text_a)
        # print(example.labels)
        # print(example.query_tokens)
        # print(example.query_labels)

        tokens = []
        labels = []
        for word, label in zip(example.text_a, example.labels):
            # print(word)
            # print(label)
            if do_lower_case:
                word=word.lower()
            word_tokens = tokenizer.tokenize([word])
            # print(word_tokens)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            if label == 'O':
                labels.extend([label] + ['O'] * (len(word_tokens) - 1))
            else:
                labels.extend([label] + ['I-BioNE'] * (len(word_tokens) - 1))


        query_tokens = []
        query_labels = []
        for query_word, query_label in zip(example.query_tokens.split(), example.query_labels.split()):
            query_word_tokens = tokenizer.tokenize([query_word])
            query_tokens.extend(query_word_tokens)
            if query_label == 'O':
                query_labels.extend([query_label] + ['O'] * (len(query_word_tokens) - 1))
            else:
                query_labels.extend([query_label] + ['I-BioNE'] * (len(query_word_tokens) - 1))

        all_labels = labels + ['O'] + query_labels

        subjects = get_entities(labels,id2label=None,markup='bio')
        all_subjects = get_entities(all_labels,id2label=None,markup='bio')

        subjects_id = []
        for subject in subjects:
            label = subject[0]
            start = subject[1]
            end = subject[2]
            subjects_id.append((label2id[label], start, end))
 
        # print(tokens)
        # print(labels)
        # print(subjects)
        # textlist = example.text_a
        # labels = example.labels
        # print('textlist:',textlist)
        # print('subjects:',subjects)
        # tokens = tokenizer.tokenize(textlist)
        # print(tokens)
        start_ids = [0] * len(all_labels)
        end_ids = [0] * len(all_labels)
        for subject in all_subjects:
            label = subject[0]
            start = subject[1]
            end = subject[2]
            start_ids[start] = label2id[label]
            end_ids[end] = label2id[label]

        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens+['[SEP]']+query_tokens) > max_seq_length - special_tokens_count:
            print('sentence:',tokens+['[SEP]']+query_tokens)
            query_tokens = tokens[: (max_seq_length - special_tokens_count - 1 - len(tokens))]
            start_ids = start_ids[: (max_seq_length - special_tokens_count)]
            end_ids = end_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        # start_ids += [0]
        # end_ids += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        # if cls_token_at_end:
        #     tokens += [cls_token]
        #     start_ids += [0]
        #     end_ids += [0]
        #     segment_ids += [cls_token_segment_id]
        # else:
        tokens = [cls_token] + tokens
        start_ids = [0]+ start_ids
        end_ids = [0]+ end_ids
        segment_ids = [cls_token_segment_id] + segment_ids



        query_tokens += [sep_token]
        start_ids += [0]
        end_ids += [0]
        segment_ids += [sequence_b_segment_id] * len(query_tokens)


        input_ids = tokenizer.convert_tokens_to_ids(tokens+query_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(tokens+query_tokens)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        # if pad_on_left:
        #     input_ids = ([pad_token] * padding_length) + input_ids
        #     input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        #     segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        #     start_ids = ([0] * padding_length) + start_ids
        #     end_ids = ([0] * padding_length) + end_ids
        # else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        start_ids += ([0] * padding_length)
        end_ids += ([0] * padding_length)

        # print(len(input_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(start_ids) == max_seq_length
        assert len(end_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens+query_tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("start_ids: %s" % " ".join([str(x) for x in start_ids]))
            logger.info("end_ids: %s" % " ".join([str(x) for x in end_ids]))
            logger.info("subjects: %s" % " ".join([str(x) for x in subjects]))
            logger.info("input_len: %s" % "".join([str(x) for x in str(input_len)]))

        features.append(InputFeature(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  start_ids=start_ids,
                                  end_ids=end_ids,
                                  subjects=subjects_id,
                                  input_len=input_len))
    return features

class CnerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "CONT", "ORG","LOC",'EDU','NAME','PRO','RACE','TITLE']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i+1)
            text_a = line['words']
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            subject = get_entities(labels,id2label=None,markup='bios')
            examples.append(InputExample(guid=guid, text_a=text_a, subject=subject))
        return examples

class BnerProcessor(DataProcessor):
    """Processor for the Biomedical ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.txt")), self._read_query_text(os.path.join(data_dir, "query_train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.txt")), self._read_query_text(os.path.join(data_dir, "query_dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.txt")), self._read_query_text(os.path.join(data_dir, "query_test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "BioNE"]

    def _create_examples(self, lines, queries, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i+1)
            text_a = line['words']
            labels = line['labels']
            query_tokens, query_labels = queries[i].split('\t')


            # print(text_a)
            # print(labels)
            # subject = get_entities(labels,id2label=None,markup='bio')  #在后面处理
            # print(subject)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels, query_tokens=query_tokens, query_labels=query_labels))
        return examples


class CluenerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "address", "book","company",'game','government','movie','name','organization','position','scene']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            labels = line['labels']
            subject = get_entities(labels,id2label=None,markup='bios')
            examples.append(InputExample(guid=guid, text_a=text_a, subject=subject))
        return examples

ner_processors = {
    "cner": CnerProcessor,
    "cluener":CluenerProcessor,
    "bner": BnerProcessor
}


