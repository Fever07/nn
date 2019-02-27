import numpy
from datetime import datetime
import pickle

def attack_generator(sess,
                     model,
                     generator,
                     adv_generator,
                     out_file,
                     save_examples=False,
                     out_dir=None,
                     orig_format='img_{0}_{1:0.3}_{2:0.3}.png',
                     adv_format='img_{0}_{1:0.3}_{2:0.3}_adv.png'):
    def map_indices(inds, iteration, batch_size):
        return inds + iteration * batch_size

    def compose_line(ind, prob, a_prob):
        f = '{0:0.6}'
        str_prob = list(map(f.format, prob))
        str_prob = ', '.join(str_prob)
        str_a_prob = list(map(f.format, a_prob))
        str_a_prob = ', '.join(str_a_prob)
        resf = '{0} - {1} - {2}\n'
        return resf.format(ind, str_prob, str_a_prob)

    file = open(out_file, 'a')
    batch_size = generator.batch_size
    for i in range(len(generator)):
        imgs, probs, labels = generator[i]

        print('Processing {0}/{1} images - {2}'.format(
            (i + 1) * batch_size,
            len(generator) * batch_size,
            datetime.now().time().strftime('%H:%M:%S')
        ))

        a_imgs = adv_generator.generate(sess, imgs, labels, verbose=False)
        a_probs = model.predict(a_imgs, batch_size=batch_size)
        
        for j, (prob, a_prob) in enumerate(zip(*(probs, a_probs))):
            line = compose_line(j, prob, a_prob)
            file.write(line)
            
    file.close()

def attack_iterative_generator(sess,
                                 model,
                                 generator,
                                 adv_generator,
                                 out_file_format,
                                 ):
    def compose_line(prob, a_prob):
        f = '{0:0.6}'
        str_prob = list(map(f.format, prob))
        str_prob = ', '.join(str_prob)
        str_a_prob = list(map(f.format, a_prob))
        str_a_prob = ', '.join(str_a_prob)
        resf = '{0} - {1}\n'
        return resf.format(str_prob, str_a_prob)

    def compose_lines(probs, a_probs):
        lines = [compose_line(*tup) for tup in zip(probs, a_probs)]
        return lines

    batch_size = generator.batch_size
    max_iter = adv_generator.max_iter
    iters = numpy.arange(1, max_iter + 1)
    out_files = []
    for it in iters:
        out_files.append(open(out_file_format.format(it), 'wb'))
    a_probs = numpy.empty(shape=[max_iter, 0, generator.num_classes], dtype=numpy.float32)
    for i in range(len(generator)):
        imgs, probs, labels = generator[i]

        print('Processing {0}/{1} images - {2}'.format(
            (i + 1) * batch_size,
            len(generator) * batch_size,
            datetime.now().time().strftime('%H:%M:%S')
        ))

        a_probs_for_iters = adv_generator.generate_and_predict(sess, imgs, labels, verbose=False)
        
        a_probs_for_iters = numpy.array(a_probs_for_iters)
        a_probs = numpy.append(a_probs, a_probs_for_iters, axis=1)

        # for file, a_probs in zip(out_files, a_probs_for_iters):
        #     lines = compose_lines(probs, a_probs)
        #     file.writelines(lines)

    for file, a_probs_for_iter in zip(out_files, a_probs):
        pickle.dump(a_probs_for_iter, file)
        file.close()
