"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.

- this will be regular BPE with Picky BPE deletions
- with counts for merges

"""

from typing import Optional, Tuple, Pattern
# note: need the fancy regex here for character classes
import regex as re
import time, json
from collections import defaultdict

from .util import frombytes, verify_dicts, frombytespair, _write_sorted_dict_intkey, _read_sorted_dict_intkey, blow_up

from .uniformbase import UniformTokenizer, merge, _write_sorted_dict, _read_sorted_dict
from .regexconstants import *

# TODO: put a limit on token length of 16 here for whitespace?

# generate sequential ID's from 0
class IndexGenerator:
    _next_index = 0

    @classmethod
    def get_next_index(cls):
        current_index = cls._next_index
        cls._next_index += 1
        return current_index

class BPEState:

    def __init__(self, prefix : str, tau: float, is_super : bool, unlocked: defaultdict[bytes,bool]):

        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """

        # what we print out for a merge, i.e. w or s for word or superword
        self.prefix = prefix

        # threshold for deletion
        self.tau = tau
    
        # two parallel lists of equal dimension
        # a list over each pre-tokenized chunk
        # the text_chunks are split into either single bytes or pre-tokenized chunks initially 
        # and merged over time
        # the corresponding text_counts are used for adjusting counts 
        # in pair_counts.  
        # The text_counts are static over time after being set up in pretokenize
        # the text_counts are always 1 when a chunk represents a document
        self.text_chunks : Optional[list[list[bytes]]] = None
        self.text_counts : Optional[list[int]] = None

        # count of each pair of adjacent tokens in self.text_chunks
        self.pair_counts : Optional[dict[tuple[bytes,bytes], int]] = None
        
        # counts of each individual tokens in self.text_chunks
        # tokens are removed from this when they hit a zero count
        # so as an invariant we should never have an entry with a zero count
        self.single_counts : Optional[dict[bytes, int]] = None

        # iteratively merge the most common pairs to create new tokens
        # the value of merges records the order index
        # with merges we want to take the merge with the smallest index first
        self.merges : Optional[dict[int,tuple[bytes,bytes]]] = {}       # int -> (bytes, bytes)
        self.deletions : Optional[dict[int, bytes]] = {}                # int -> bytes

        self.unlocked : defaultdict[bytes, bool] = unlocked  # can we use this in a pair

        # number of single token chunks, for stats
        self.whole_words = 0 

        # a static set of single bytes, for computing the total number of single bytes in stats
        self.single_bytes = set([bytes([idx]) for idx in range(256)])

        # we will vary things a bit between words and superwords
        self.is_super = is_super

        # TODO: hardcoded, pass this in
        # must have at least one letter, then can also have spaces, underscores or apostrophes
        self.merge_pattern = re.compile(rb"^[ _'a-zA-Z]*[a-zA-Z][ _'a-zA-Z]*$")

    # keep separate, we can check they both have letters
    def can_merge(self, left : bytes, right : bytes) -> bool:
        return (not self.is_super) or (bool(re.match(self.merge_pattern, left)) and bool(re.match(self.merge_pattern, right)))

    # just called on super, is a single string compatible with self.merge_pattern
    def could_merge(self, tok : bytes) -> bool:
        return bool(re.match(self.merge_pattern, tok))

    # work at a chunk level
    # where a document is stored as a list of chunks
    # TODO: hardcoded max_bytes fix me!
    def pretokenize(self, filepath : str, num_lines : int, compiled_pattern : Pattern[str], max_bytes = 1000000000) -> None:

        start_time = time.time()

       # get counts of pre-tokenized chunks for each document
        chunk_tally : dict[str, int] = {}

        total_bytes = 0
        total_chars = 0

        # for regular words tally things up
        if not self.is_super:

            with open(filepath) as f:
                for i in range(num_lines):
                    if i % 10000 == 0:
                        print("document", i, time.time() - start_time, total_chars, total_bytes)
                    line = f.readline()
                    text = json.loads(line.rstrip())["text"]

                    # split the text up into text chunks
                    for chunk in re.findall(compiled_pattern, text):
                        chunk_tally[chunk] = chunk_tally.get(chunk, 0) + 1

                    total_chars += len(text)
                    total_bytes += len(text.encode("utf-8"))

                    if total_bytes >= max_bytes:
                        print('at max_bytes', i, max_bytes, total_chars, total_bytes)
                        break

            # lets make parallel list of chunks and counts before we split up the chunks
            # sort descending by count for neatness, TODO: can take out later
            cnt_chk = sorted([(cnt,chk) for chk,cnt in chunk_tally.items()], reverse=True)

            print("number of pre-toknization chunks:", len(cnt_chk))
            print("top 10 pre-tokenization chunks:")
            for j in range(10):
                print(j, cnt_chk[j])
            print()

            # store in tokenizer
            self.text_counts, self.text_chunks = zip(*cnt_chk)
            
            # convert string to bytes, and split into single bytes to get started
            self.text_chunks = [[bytes([b]) for b in ch.encode("utf-8")] for ch in self.text_chunks]

        else:

            # for super words, just get all the the text chunks, and leave them as documents
            self.text_chunks = []

            with open(filepath) as f:
                for i in range(num_lines):
                    if i % 10000 == 0:
                        print("document", i, time.time() - start_time, total_chars, total_bytes)
                    line = f.readline()
                    text = json.loads(line.rstrip())["text"]

                    # split the text up into text chunks
                    # and just convert them to bytes
                    document = [chunk.encode("utf-8") for chunk in re.findall(compiled_pattern, text)]
                    self.text_chunks.append(document)

                    total_chars += len(text)
                    total_bytes += len(text.encode("utf-8"))

                    if total_bytes >= max_bytes:
                        print('at max_bytes', i, max_bytes, total_chars, total_bytes)
                        break

            # all the counts are 1 here
            self.text_counts = [1]*len(self.text_chunks)

        # should always stay in parallel
        assert len(self.text_chunks) == len(self.text_counts)

        total_counts = sum(self.text_counts)

        print("pretokenization time:",time.time() - start_time, len(self.text_counts), total_counts)


    def print_tokenization(self):

        print()
        print("final tokenization:")
        for tokens, cnt in zip(self.text_chunks, self.text_counts):

            # I'm curious about ones with a single byte in them
            minlen = min([len(tok) for tok in tokens])
            output = " ".join([frombytes(tok) for tok in tokens])

            print("!", minlen, len(tokens), output, cnt)

    #############################################################################

    # for this to be quick, we have to rely on the fact that most pretokens are skipped entirely
    def get_stats(self,
                  tokens : list[bytes], 
                  counts : Optional[dict[Tuple[bytes,bytes],int]],
                  multiplier : int,  # +cnt or -cnt depending
                  ) -> dict[Tuple[bytes,bytes],int]:
        """
        Given a list of bytes object tokens, 
        return a dictionary of counts of consecutive pairs
        Example: [b'a', b'b', b'c', b'a', b'b'] -> {(b'a', b'b'): 2, 
                                                    (b'b', b'c'): 1, 
                                                    (b'c', b'a'): 1}
        Optionally allows an existing dictionary of counts to update
        """
        counts = {} if counts is None else counts
        prev_pair = None
        for pair in zip(tokens, tokens[1:]): # iterate consecutive elements
            pairunlocked = self.unlocked[pair[0]] and self.unlocked[pair[1]]
            sameasprevious = pair[0] == pair[1] and pair == prev_pair
            if pairunlocked and not sameasprevious:
                counts[pair] = counts.get(pair, 0) + multiplier
                prev_pair = pair
            else:
                prev_pair = None # do the next one but skip this one

        return counts


    # return the pair counts without any side effects
    def _calc_pair_counts(self):

        pair_counts : dict[Tuple[bytes,bytes], int] = {}
        for tokens, cnt in zip(self.text_chunks, self.text_counts):
            # passing in stats will update it in place, adding up counts, 
            # incrementing each occurence by cnt
            # be sure to use get_stats here to handle overlapping tokens correctly
            self.get_stats(tokens, pair_counts, cnt)

        return pair_counts

    # get initial pair_counts, which we'll keep updating
    def initial_pair_counts(self):

        self.pair_counts = self._calc_pair_counts()

    # double check our dynamic single counts are being updated correctly
    def verify_pair_counts(self):
        from_scratch = self._calc_pair_counts()
        verify_dicts(from_scratch, self.pair_counts)

        # we should have the invariant that every element of self.paircounts is unlocked
        for pair in self.pair_counts:
            for i in [0,1]:
                if not self.unlocked[pair[i]]:
                    print("not unlocked:", pair[i], pair)
                assert self.unlocked[pair[i]]
            
    # compute single counts from scratch, and return them
    def _calc_single_counts(self, verbose=False) -> dict[bytes, int]:

        single_counts : dict[bytes, int] = {}
        for tokens, cnt in zip(self.text_chunks, self.text_counts):
            for tok in tokens:
                single_counts[tok] = single_counts.get(tok, 0) + cnt

        return single_counts

    # set up the single token counts, which also update over time
    def initial_single_counts(self):
        self.single_counts = self._calc_single_counts(verbose=True)

    # double check our dynamic single counts are being updated correctly
    def verify_single_counts(self) -> None:
        from_scratch = self._calc_single_counts()
        verify_dicts(from_scratch,  self.single_counts)

    def _calc_whole_words(self):
        whole_words = 0
        for tokens, cnt in zip(self.text_chunks, self.text_counts):
            if len(tokens) == 1:
                whole_words += cnt
        return whole_words

    def initial_whole_words(self):
        self.whole_words = self._calc_whole_words()

    def verify_whole_words(self):
        from_scratch = self._calc_whole_words()

        if self.whole_words != from_scratch:
            print("debug 2:", self.whole_words, from_scratch)
        assert self.whole_words == from_scratch

    
    # verify it all
    def verify_state(self) -> None:
        self.verify_pair_counts()
        self.verify_single_counts()
        self.verify_whole_words()

    # compute our initial state from self.text_chunks and self.text_counts
    def initial_counts(self) -> None:
        self.initial_pair_counts()
        self.initial_single_counts()
        self.initial_whole_words()

    # a specialized version 
    # this shouldn't change state
    def choose_best_pair(self, vocablist) -> Optional[Tuple[bytes,bytes]]:
        
        max_cnt = -1
        max_cnt_pair = None
                
        # find the k pairs with the highest max_count
        # we have the invariant that both element of pair are unlocked
        for pair, cnt in self.pair_counts.items():

            # break ties deterministically
            # avoid merging a pair that is already in the vocabulary
            # it seems like this can happen with the combination of 
            # merges and super merges
            # TODO: is the final check still necessary?
            if ((cnt > max_cnt) or (cnt == max_cnt and pair < max_cnt_pair)) and (pair[0] + pair[1] not in vocablist):
                max_cnt = cnt 
                max_cnt_pair = pair

        # these should have been unlocked to end up in pair_count
        if max_cnt_pair is not None:
            assert self.unlocked[max_cnt_pair[0]]
            assert self.unlocked[max_cnt_pair[1]]

        # this is None if nothing left to merge
        return max_cnt_pair, max_cnt

    def merge_and_update(self, max_pair : Tuple[bytes,bytes]) -> int:

        # unpack once 
        first, second = max_pair

        # these should have been unlocked
        assert self.unlocked[first]
        assert self.unlocked[second]

        # the combined changes of pair counts over all text_chunks
        # this way we can track the number of changing values from this merge
        # and only update those values in 
        overall_change = {}

        # how many of the merged count were actually created
        # need this to be accurate even with overlapping merges
        total_merge_cnt = 0
        whole_word_increase = 0
        
        # is there a chunk that is a single token after merging max_pair
        new_unlocked = None

        for tokens, cnt in zip(self.text_chunks, self.text_counts):

            # this is usually False, so is a big speedup
            possible = False

            # does `first` appear next to `second` in the list?
            # I'm hoping this is faster for long lists of tokens
            # since only need one pass throught the list
            for i in range(len(tokens) - 1):  # Iterate up to the second-to-last element
                if tokens[i] == first and tokens[i + 1] == second:
                    possible = True
                    break

            if possible:

                # just find the deltas in the current text_chunk
                # handling overlapping tokens correctly
                local_delta = self.get_stats(tokens, counts=None, multiplier=-cnt)

                # when `possible` should mean it is in there
                if max_pair not in local_delta:
                    print("debug 4:", frombytespair(max_pair))
                    assert max_pair in local_delta

                before_len = len(tokens)

                # then merge each chunk independently
                # replace all occurrences of pair in tokens with the combined
                # the ending number may be less than the number of pairs
                # this is the unweighted number of merges
                newtokens, merge_cnt = merge(tokens, max_pair)
                # use the slice notation so the change persists outside of the iteration
                tokens[:] = newtokens

                # this tokens appears cnt times in the pre-tokenization                 
                total_merge_cnt += merge_cnt*cnt

                # and finally, increment local_delta for each ending pair
                self.get_stats(tokens, counts=local_delta, multiplier=cnt)

                # copy over the local ones to the overall change
                # ignore the ones that cancelled out, which should be many 
                for pair, delta in local_delta.items():
                    if delta != 0:
                        overall_change[pair] = overall_change.get(pair, 0) + delta

                        if overall_change[pair] == 0:
                            del overall_change[pair]

                if len(tokens) == 1 and (before_len > 1):
                    whole_word_increase += cnt

                    # only merge space words
                    new_unlocked = first + second                        

                    # we'll return this too so we can affect superwords
                    self.unlocked[new_unlocked] = True
                    
        # add the deltas to our self.pair_counts
        for pair, delta in overall_change.items():
            if delta != 0:
                self.pair_counts[pair] = self.pair_counts.get(pair, 0) + delta

                # delete when we have a 0 entry
                if self.pair_counts[pair] == 0:
                    del self.pair_counts[pair]

        # should no longer have max_pair
        if max_pair in self.pair_counts:
            print("debug 5:", frombytespair(max_pair), self.pair_counts[max_pair])
        assert max_pair not in self.pair_counts

        # now also should update the 3 single counts that changed because of the merge
        # now with deletions and also super merges we can get things another way
        # TODO: still valid?
        merged_pair = first + second
        if merged_pair in self.single_counts:
            print("warning 1:", frombytes(merged_pair), frombytespair(max_pair), "was", self.single_counts[merged_pair], total_merge_cnt)
        # assert merged_pair not in self.single_counts
        self.single_counts[merged_pair] = self.single_counts.get(merged_pair, 0) + total_merge_cnt

        # and decrease the count of first and second by the same amount
        # still works if first == second
        self.single_counts[first] -= total_merge_cnt
        self.single_counts[second] -= total_merge_cnt

        if self.single_counts[first] == 0:
            del self.single_counts[first]

        # we don't want to delete it twice if first == second
        if first != second and self.single_counts[second] == 0:
            del self.single_counts[second]

        self.whole_words += whole_word_increase
                
        # the single counts decreased by total_merge_cnt
        return total_merge_cnt, new_unlocked

    # delete a particular token, splitting all occurences into single bytes 
    # update counts accordingly
    # TODO: should this potentially change the locked status of bad_token
    # or do we not deleted locked tokens
    def delete_and_update(self, bad_token : bytes) -> None:
        
        # if ios==1 then we already deleted them all
        # but still need to delete from the vocab
        expected_cnt = self.single_counts.get(bad_token, 0)

        # the combined changes over all text_chunks
        # this way we can track the number of changing values from this deletion
        # and only update those values that changed
        overall_change = {}

        # how many of the merged count were actually created
        # need this to be accurate even with overlapping merges
        total_delete_cnt = 0
        # the number of whole words will be non-positive
        whole_word_increase = 0

        for tokens, cnt in zip(self.text_chunks, self.text_counts):
            # compute the change in the list `tokens`

            # this is usually False, so is a big speedup
            if bad_token in tokens:

                # just find the deltas in the current text_chunk
                # handling overlapping tokens correctly
                # this is the change in *pairs*
                local_delta = self.get_stats(tokens, counts=None, multiplier=-cnt)

                before_len = len(tokens)

                # then merge each chunk independently
                # replace all occurrences of pair in tokens with the combined
                # the ending number may be less than the number of pairs
                # this is the unweighted number of merges
                newtokens, deletions = blow_up(tokens, bad_token)
                # use the slice notation so the change persists outside of the iteration
                tokens[:] = newtokens

                # tally how many merges we did                 
                total_delete_cnt += deletions*cnt

                # and finally, increment local_delta for each ending pairs
                self.get_stats(tokens, counts=local_delta, multiplier=cnt)
                # copy over the local ones to the overall change
                # ignore the ones that cancelled out, which should be many 
                for pair, delta in local_delta.items():
                    if delta != 0:
                        overall_change[pair] = overall_change.get(pair, 0) + delta

                        if overall_change[pair] == 0:
                            del overall_change[pair]

                # did we go from a single token to more
                if before_len == 1 and len(tokens) > 1:
                    whole_word_increase -= cnt

        if expected_cnt != total_delete_cnt:
            print("debug 8:", frombytes(bad_token), expected_cnt, total_delete_cnt, cnt)
        assert expected_cnt == total_delete_cnt

        # adjust the whole words too 
        self.whole_words += whole_word_increase  #  a negative increase

        for pair, delta in overall_change.items():
            if delta != 0:
                self.pair_counts[pair] = self.pair_counts.get(pair, 0) + delta

                # delete when we have a 0 entry
                if self.pair_counts[pair] == 0:
                    del self.pair_counts[pair]

        # if ios == 1, then bad_token was deleted already from self.single_counts
        # e.g. deleting Ġsugg after the merger of Ġsugg est
        if bad_token in self.single_counts:
            if self.single_counts[bad_token] != total_delete_cnt:
                print("debug 9:", frombytes(bad_token), self.single_counts[bad_token], total_delete_cnt)
            self.single_counts[bad_token] -= total_delete_cnt
            # only keep single_counts with nonzero counts, so delete this
            assert self.single_counts[bad_token] == 0
            del self.single_counts[bad_token]

            # increase the single counts of the single bytes 
            # if it has repeated bytes we'll just do them separately
            # in the ios==1 case we have total_delete_cnt==0, so this is a no op
            # can be inside the if statement
            for tok in [bytes([b]) for b in bad_token]:
                before = self.single_counts.get(tok, 0)
                self.single_counts[tok] = before + total_delete_cnt
                # print("singles:", tok, before, total_delete_cnt)

        if bad_token in self.deletions.values():
            print("debug 10", frombytes(bad_token))

        # save the deletion event
        self.deletions[IndexGenerator.get_next_index()] = bad_token


    # update pairwise counts according to the newly unlocked token
    # update counts accordingly
    # note that this doesn't actually change anything in self.text_chunks
    # just values in self.pair_counts
    def unlock_and_update(self, new_unlocked : bytes) -> None:
        
        # if the newly unlocked token isn't something that can be 
        # in a supermerge, then nothing to do
        # we'll leave this as locked
        # thus everything in self.unlocked should have could_merge == True
        if not self.could_merge(new_unlocked):
            return 

        # mark it as unlocked
        self.unlocked[new_unlocked] = True

        # the combined changes over all text_chunks
        # this way we can track the number of changing values from this deletion
        # and only update those values that changed
        for tokens, cnt in zip(self.text_chunks, self.text_counts):
            # compute the change in the list `tokens`

            n = len(tokens)

            i = 0
            while i < n:

                tok = tokens[i]

                if tok == new_unlocked:
                                    
                    # now find out how many repeated copies of tok appear after this
                    k = i 
                    while (k+1 < n) and tokens[k+1] == new_unlocked:
                        k += 1

                    # this gives us one or more copies of tok from i to k
                    assert tokens[k] == new_unlocked

                    copies = (k-i+1)   # upper - lower + 1
                    assert copies >= 1

                    if (copies % 2 == 0):
                        pair = (tok,tok)
                        self.pair_counts[pair] = self.pair_counts.get(pair, 0) + cnt*(copies // 2)  # probably cnt is 1
                    else:
                        # if just one, then won't be making any of these 
                        if copies > 1:
                            pair = (tok,tok)
                            # ignore the odd one, and then also have n-1 intervals 
                            self.pair_counts[pair] = self.pair_counts.get(pair, 0) + cnt*((copies-1) // 2)  # probably cnt is 1

                    if i-1 >= 0:
                        
                        if self.unlocked[tokens[i-1]]:

                            # I think everything unlocked in the super should satisfy could_merge
                            assert self.could_merge(tokens[i-1])
                                                    
                            pair = (tokens[i-1],tok)
                            # I'm assuming all our cnt are 1 here, but I guess we might as well use it
                            self.pair_counts[pair] = self.pair_counts.get(pair, 0) + cnt  

                    if k+1 < len(tokens):

                        if self.unlocked[tokens[k+1]]:
                                
                            # I think everything unlocked in the super should satisfy could_merge
                            assert self.could_merge(tokens[k+1])

                            pair = (tok,tokens[k+1])
                            # I'm assuming all our cnt are 1 here, but I guess we might as well use it
                            self.pair_counts[pair] = self.pair_counts.get(pair, 0) + cnt  
                        
                    # skip over the runs of tok
                    i = k+1

                else:
                    # advance to the next one
                    i += 1 
                        
            
    # TODO: now need to decide this based on a whole doc, 
    # so we can support supermerges
    # TODO: need to update this for supermerges
    # def min_merge_or_deletion(self, tokens : list[bytes]) -> Optional[Tuple[bytes,bytes]]:
    #     """
    #     Given a list of bytes object tokens, 
    #     return the merge or deletion with the lowest index
    #     or None if none are found
    #     """
    #     min_pair = None             # returns None if nothing to merge 
    #     min_cnt = 1e9  # initialize to a value larger than any index

    #     for pair in zip(tokens, tokens[1:]): # iterate consecutive elements
    #         if pair in self.merges and self.merges[pair] < min_cnt:
    #             min_cnt = self.merges[pair]
    #             min_pair = pair

    #     # and now check if there is a deletion with a lower count
    #     for tok in tokens:
    #         if tok in self.deletions and self.deletions[tok] < min_cnt:
    #             min_cnt = self.deletions[tok]
    #             min_pair = (tok,)  # return a tuple with a single element here

    #     return min_pair

    # how many tokens are single bytes in the current tokenization
    def get_single_byte_cnt(self):
        return sum([self.single_counts.get(tok,0) for tok in self.single_bytes])

    # do a merge of best_pair, followed by a potential deletion
    def merge_and_delete(self, best_pair, i, vocablist, start_time, verbose):

            left, right = best_pair

            assert self.unlocked[left], frombytes(left) + "," + str(self.is_super) + frombytespair(best_pair)
            assert self.unlocked[right], frombytes(right) + "," + str(self.is_super) + frombytespair(best_pair)
            
            # the c_ab and best_loss we used in choose_best_pair 
            # to select best_pair may have changed by now
            # due to previous merges in batch, so recompute
            c_ab = self.pair_counts[best_pair]
            # look up the corresponding single values to for output
            c_a = self.single_counts[left]
            c_b = self.single_counts[right]

            # compute the Intersection over Self metrics from 
            # compute before merging
            ios_a = c_ab/c_a
            ios_b = c_ab/c_b

            # do the merges while updating pair_counts as appropriate
            start_merge = time.time()
            total_merge_cnt, new_unlocked = self.merge_and_update(best_pair)

            # did we merge what we expected to?
            if c_ab != total_merge_cnt:
                print("debug 11:", c_ab, total_merge_cnt, frombytespair(best_pair))
            assert c_ab == total_merge_cnt

            # create new token
            merged_tok = left + right
            if merged_tok in vocablist:
                print("warning 2:", frombytes(merged_tok), frombytespair(best_pair), "in vocab")
            # assert merged_tok not in self.vocablist

            # save the merge and vocab with next available index
            self.merges[IndexGenerator.get_next_index()] = best_pair
            merge_type = self.prefix

            vocablist.append(merged_tok) # is bigger than merges due to single bytes

            merge_time = time.time() - start_merge

            if verbose:

                left_tok = frombytes(left)
                right_tok = frombytes(right)

                if new_unlocked is None:
                    nu = ""
                else:
                    nu = frombytes(new_unlocked)

                output = ["*", i+1, len(vocablist), merge_type, \
                        left_tok, right_tok, c_ab, c_a, c_b, round(ios_a, 5), round(ios_b, 5), round(time.time()-start_time,5), \
                        len(self.pair_counts), \
                        self.get_single_byte_cnt(), self.whole_words, nu]
                print("\t".join([str(x) for x in output]))

            # now see if we need delete any tokens
            start_delete = time.time()
            for (ios, tok, direction) in [(ios_a, left, "left"), (ios_b, right, "right")]:
                # don't delete a single byte or a super merge
                # note if ios == 1 then we already deleted all occurences
                if (len(tok) > 1) and (ios >= self.tau):

                    if new_unlocked is None:
                        nu = ""
                    else:
                        nu = frombytes(new_unlocked)

                    output = ["*", i+1, len(vocablist), self.prefix + "d", \
                        frombytes(tok), direction, c_ab, c_a, c_b, round(ios, 5), round(self.tau, 5), round(time.time()-start_time,5), \
                        len(self.pair_counts), \
                        self.get_single_byte_cnt(), self.whole_words, nu]
                    
                    print("\t".join([str(x) for x in output]))

                    if tok in self.deletions.values():
                        print("warning: deleting something twice", frombytes(tok))

                    self.delete_and_update(tok)
                            # and delete from vocab
                    vocablist  = [t for t in vocablist if t != tok]  
                        
            delete_time = time.time() - start_delete

            return merge_time, delete_time, new_unlocked



class PickyBPE(UniformTokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()

        self.pattern = ULTIMATE_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        
        # TODO: these need more testing
        self.special_tokens = {}    # str -> int
        self.inverse_special_tokens = {}  # int -> str

        # since we'll be deleting things, don't assign the id values until the end
        # so make it a list for now during training
        self.vocablist : Optional[list[bytes]] = [bytes([idx]) for idx in range(256)]

        # and then it gets converted after training to a dict, 
        # once we're done with deletions
        self.vocab : Optional[dict[bytes,int]] = None

        # we'll have two of these for words and superwords
        always_unlocked : defaultdict[bytes, bool] = defaultdict(lambda: True)  # everything is unlocked for the words

        self.words_state = BPEState('w', 0.9, is_super = False, unlocked=always_unlocked)

    # verify indices are correct
    def verify_indicies(self) -> None:

        indices = sorted(list(self.words_state.merges.keys()) + \
                         list(self.words_state.deletions.keys()))

        # should be an ordered list of indices
        for i, ind in enumerate(indices):

            # about to die so dump them
            if i != ind:

                print("debug 3:", i, ind)
                print("word merges:")
                for idx, pair in self.words_state.merges.items():
                    print(idx, frombytespair(pair))
                print()
                print("word deletions:")
                for idx, tok in self.words_state.deletions.items():
                    print(idx, frombytes(tok))
                print()

            assert i == ind


    # file_path : the location of the MiniPile *.jsonl file
    # num_lines : how many documents to read from MiniPile, out of 1M
    # vocab_size : when to stop
    # method : "count", "l2", "l1", "tfidf", "minslack" TODO: fix
    # min_cnt : minimum support, also stop when all are below this
    # verbose : should we print output?
    def train(self, 
              filepath : str, 
              outprefix : str,
              num_lines : int, 
              vocab_size: int, 
              recalc : int,  # how many iterations do we recompute from scratch
              verbose:bool = True) -> None:

        assert vocab_size >= 256

        start_overall = time.time()

        # timing splits
        total_pretok = 0.0
        total_ic = 0.0
        total_max_value = 0.0
        total_merge = 0.0
        total_delete = 0.0
        total_unlocked = 0.0
        total_verify = 0.0

        # set up self.text_counts and self.text_chunks
        start_pretok = time.time()
        self.words_state.pretokenize(filepath, num_lines, self.compiled_pattern)
        total_pretok = time.time() - start_pretok
        
        # set up our initial counts
        start_ic = time.time()
        self.words_state.initial_counts()
        # must have set up the two previous first
        total_ic = time.time() - start_ic
        
        if verbose:

            header = ["*", "i", "vocab", "merge", \
                      "left", "rght", "c_ab", "c_a", "c_b", "ios_a", "ios_b", "time", \
                      "pairs", "single_bytes", "whole_words", "new_unlocked"]
            
            print("\t".join(header))

        i = 0
        while len(self.vocablist) < vocab_size:

            # we have to stop when everything is merged into a single token
            # hence there are no pairs with a count
            if len(self.words_state.pair_counts) == 0:
                print("only single element chunks", i)
                break

            # time each iteration between every iterations
            start_time = time.time()

            start_max = time.time()
            best_pair_words, best_cnt_words = self.words_state.choose_best_pair(self.vocablist)

            bpw = "none"
            if best_pair_words is not None:
                bpw = frombytes(best_pair_words[0] + best_pair_words[1])

            print("best pair:", bpw, best_cnt_words)

            total_max_value += (time.time() - start_max)
            
            # should break out of while loop if didn't find anything
            # because of min_cnt 
            if best_pair_words is None:
                print("best is None for words")
                break

            ########### 

            #this side effects self.vocablist as necessary
            merge_time, delete_time, _ = self.words_state.merge_and_delete(best_pair_words, i, self.vocablist, start_time, verbose)

            total_merge += merge_time
            total_delete += delete_time

            # verify dynamic stuff from scratch every so often
            # TOD: magic number here
            if i % recalc == 0 and i > 0:
                verify_start = time.time()
                self.words_state.verify_state()
                total_verify += time.time() - verify_start
                self.verify_indicies()

            i += 1

            # save intermediate output
            if len(self.vocablist) % 8192 == 0:  #  100 == 0:

                self.vocab = { tok : ind for (ind, tok) in enumerate(self.vocablist)}
                overall_time = time.time() - start_overall

                print(":len(vocab):", len(self.vocab))
                print(":single_counts:", len(self.words_state.single_counts))
                print(":pair_counts:", len(self.words_state.pair_counts))
                print(":single_byte_cnt:", self.words_state.get_single_byte_cnt())
                print(":whole_words:", self.words_state.whole_words)
                print(":merges:", len(self.words_state.merges))
                print(":deletions:", len(self.words_state.deletions))
                print(":training time breakdown")
                print(":total_pretok:", total_pretok)
                print(":total_initalize_counts:", total_ic)
                print(":total_max_value:", total_max_value)
                print(":total_merge:", total_merge)
                print(":total_delete:", total_delete)
                print(":total_verify:", total_verify)
                print(":overall_time:", overall_time)
                print(":missing:", overall_time - total_pretok - total_ic - total_max_value - total_merge - total_delete - total_verify)

                # outprefix = f"./models/fastersuper_{num_lines}_{len(self.vocablist)}"
                print(":outprefix:", outprefix + "_" + str(len(self.vocablist)))
                self.save(outprefix + "_" + str(len(self.vocablist)))

        # training is done here
        # now finally convert to the vocab, since we're done with deletions
        self.vocab = { tok : ind for (ind, tok) in enumerate(self.vocablist)}

        # finally compute the inverse vocabulary at the end for decoding
        self.inv_vocab = { ind : tok for (ind,tok) in enumerate(self.vocablist)}  # int -> bytes

        # make sure we don't accidentially use the list form now
        self.vocablist = None 

        # write our vocab with counts
        # TODO: have logic for supermerge vocab words too
        print()
        print("vocab:")
        for tok, index in self.vocab.items():
            print("+", index, frombytes(tok), self.words_state.single_counts.get(tok, 0))
        print()

        # this is probably too big now
        # self.print_tokenization()

        overall_time = time.time() - start_overall

        print(":len(vocab):", len(self.vocab))
        print(":single_counts:", len(self.words_state.single_counts))
        print(":pair_counts:", len(self.words_state.pair_counts))
        print(":single_byte_cnt:", self.words_state.get_single_byte_cnt())
        print(":whole_words:", self.words_state.whole_words)
        print(":merges:", len(self.words_state.merges))
        print(":deletions:", len(self.words_state.deletions))
        print(":training time breakdown")
        print(":total_pretok:", total_pretok)
        print(":total_initalize_counts:", total_ic)
        print(":total_max_value:", total_max_value)
        print(":total_merge:", total_merge)
        print(":total_delete:", total_delete)
        print(":total_verify:", total_verify)
        print(":overall_time:", overall_time)
        print(":missing:", overall_time - total_pretok - total_ic - total_max_value - total_merge - total_delete - total_verify)

    # TODO: test this
    def register_special_tokens(self, special_tokens : list[str]):
        # special_tokens is a list of str
        # example: {"<|endoftext|>": 100257}

        # call this *after* training
        if self.vocab is None or len(self.vocab) == 0:
            print("please call register_special_tokens after training")
        assert self.vocab is not None and (len(self.vocab) > 0)

        self.special_tokens = {}
        for tok in special_tokens:
            self.special_tokens[tok] = len(self.vocab) + len(self.special_tokens)

        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        # use the new format
        model_file = file_prefix + ".model"
        with open(model_file, 'wt') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("word v1\n")
            # lets json encode this now since might have \n 
            f.write(json.dumps(self.pattern) + "\n")
            _write_sorted_dict(self.vocab, f, ispair=False, isstr=False)
            # TODO: think this through
            _write_sorted_dict(self.special_tokens, f, ispair=False, isstr=True) # these are strings
            _write_sorted_dict_intkey(self.words_state.merges, f, ispair=True, isstr=False)  # these have int as the key, since tokens might not be unique          
            _write_sorted_dict_intkey(self.words_state.deletions, f, ispair=False, isstr=False)


    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        with open(model_file, 'rt', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "word v1"

            # read the json encoded pattern
            self.pattern = json.loads(f.readline().strip())
            self.compiled_pattern = re.compile(self.pattern)

            self.vocab = _read_sorted_dict(f, ispair=False, isstr=False)
            self.special_tokens = _read_sorted_dict(f, ispair=False, isstr=True)  # these are strings, why?
            self.words_state.merges  = _read_sorted_dict_intkey(f, ispair=True, isstr=False)
            self.words_state.deletions  = _read_sorted_dict_intkey(f, ispair=False, isstr=False)

            # make sure indices are consistent across these four
            self.verify_indicies()  

            # compute the inverses 
            self.inv_special_tokens = { v : k for (k,v) in self.special_tokens.items()}
            self.inv_vocab = { v:k for (k,v) in self.vocab.items()}        

    def decode(self, ids : list[int]) -> str:
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.inv_vocab:
                part_bytes.append(self.inv_vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        # won't always decode to a string
        # TODO: return a bytes instead?
        text = text_bytes.decode("utf-8", errors="replace")
        return text

