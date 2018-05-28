from lang_model import *

def lmda_smoothing(prob):

    lmda = 1e-08
    prob = prob + lmda
    prob[len(lm.dr.char_to_num)] += lmda * (V - len(lm.dr.char_to_num) -1)
    prob /= np.sum(prob)
    return prob


def compute_perplexity(test_str, lm, smooth=False):
    user_input_chars = [c for c in test_str]

    next_c, next_h = np.zeros((1, lm.h_dim)), np.zeros((1, lm.h_dim))
    first_char = START_CHAR  # first is always start
    char_bits = np.array(data_reader.char_to_bit(first_char))
    char_bits = char_bits.reshape((1, 1, 32))

    # get the initial predictions given the start of a sequence
    y_pred, next_c, next_h = lm.sess.run([lm.y_hat, lm.infer_state, lm.infer_output],
                                           feed_dict={lm.X_infer: char_bits, lm.init_c: next_c,
                                                      lm.init_h: next_h, lm.keep_prob: 1.0})
    log_sum = 0.0
    i = 0
    L = len(user_input_chars)/2
    while i < len(user_input_chars):
        if user_input_chars[i] == 'o':  # observation
            next_char = user_input_chars[i + 1]

            char_bits = np.array(data_reader.char_to_bit(next_char))
            char_bits = char_bits.reshape((1, 1, 32))

            # this character's prob is found from previous time step's probability distribution
            y_pred = y_pred.flatten()
            if smooth:
                y_pred = lmda_smoothing(y_pred)

            if next_char not in lm.dr.char_to_num:
                unk_prob = y_pred[len(lm.dr.char_to_num)]
                y_prob = unk_prob / (V - len(lm.dr.char_to_num))
            else:
                y_prob = y_pred[lm.dr.get_char_to_num(next_char)]

            log_sum += math.log(y_prob, 2)

            # get the new predictions given the current observation
            y_pred_new, next_c, next_h = lm.sess.run([lm.y_hat, lm.infer_state, lm.infer_output],
                                                       feed_dict={lm.X_infer: char_bits, lm.init_c: next_c,
                                                                  lm.init_h: next_h, lm.keep_prob: 1.0})
            y_pred = y_pred_new
            i += 2

    # Include the STOP character for computing perplexity
    y_pred = y_pred.flatten()
    if smooth:
        y_pred = lmda_smoothing(y_pred)
    y_prob = y_pred[lm.dr.get_char_to_num(STOP_CHAR)]

    log_sum += math.log(y_prob, 2)
    mean_log_sum = (1.0/(L+1))*log_sum

    perp = np.power(2, -mean_log_sum)
    print('Perplexity = {}'.format(perp))


if __name__=='__main__':
    lm = LangModel(X_dim=32, h_dim=256, max_epoch=10, batch_size=128,  keep_param = 0.5)
    run_id = str(input("enter a run id: "))
    lm.load(10, run_id)

    with open('tests/test5.txt', 'r') as f:
        test_str = f.readline().rstrip()

    compute_perplexity(test_str, lm,smooth=True)
