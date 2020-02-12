import numpy as np

def deepfool_attack(sess, x, predictions, grads, imgs, n_classes,
                    overshoot, max_iter, clip_min, clip_max):
    """
    :param sess: TF session
    :param x: The input placeholder
    :param predictions: The model's sorted symbolic output of probabilities
    :param grads: Symbolic gradients of the top nb_candidate classes, produced
                 from gradient_graph
    :param imgs: Numpy array with images
    :param n_classes: Number of classes predicted by model
    :param overshoot: A termination criterion to prevent vanishing updates
    :param max_iter: Maximum number of iteration for DeepFool
    :param clip_min: Minimum value for components of the example returned
    :param clip_max: Maximum value for components of the example returned
    :return: Adversarial examples
    """
     
    current = np.argmax(sess.run(predictions, feed_dict={x: imgs}), axis=1)
    original = current

    adv_x = imgs
    w = np.zeros(imgs.shape[1:])
    r_tot = np.zeros(imgs.shape)
    lower_bounds = clip_min - imgs
    upper_bounds = clip_max - imgs
    
    iteration = 0
    while (np.any(current == original) and iteration < max_iter):
        gradients = sess.run(grads, feed_dict={x: adv_x})
        predictions_val = sess.run(predictions, feed_dict={x: adv_x})
        for idx in range(imgs.shape[0]):
            k0 = original[idx]
            pert = np.inf
            if current[idx] != original[idx]:
                continue
            for k in set(range(0, n_classes)).difference([k0]):
                w_k = gradients[idx, k, ...] - gradients[idx, k0, ...]
                f_k = predictions_val[idx, k] - predictions_val[idx, k0]
                pert_k = (abs(f_k) + 0.00001) / np.linalg.norm(w_k)
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
            r_i = pert * w / np.linalg.norm(w)
            r_tot[idx, ...] = r_tot[idx, ...] + r_i
            # r_tot = np.clip(r_tot, lower_bounds, upper_bounds)
            # print('Iteration: {}, idx: {}, norm: {}'.format(iteration, idx, np.max(np.abs(r_tot[idx]))))

        adv_x = np.clip(imgs + r_tot, clip_min, clip_max)
        current = np.argmax(sess.run(predictions, feed_dict={x: adv_x}), axis=1)
        iteration = iteration + 1

    adv_x = np.clip(imgs + (1 + overshoot) * r_tot, clip_min, clip_max)
    return adv_x


if __name__ == '__main__':
    import tensorflow as tf

    np.random.seed(42)
    tf.set_random_seed(42)

    tf.keras.backend.clear_session()
    sess = tf.keras.backend.get_session()

    inp = tf.keras.layers.Input(shape=[20])
    out = tf.keras.layers.Dense(2)(inp)

    model = tf.keras.models.Model(inputs=[inp], outputs=[out])
    model.compile(tf.train.AdamOptimizer(0.001), loss='mse')

    X = np.random.uniform(0, 1, [200, 20])
    W = np.random.uniform(0, 1, [20, 2])
    y = X @ W
    X_train = X[:160]
    X_test = X[160:]
    y_train = y[:160]
    y_test = y[160:]

    model.fit(X_train, y_train, epochs=100)
    model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)

    grads = [tf.gradients(model.output[:, i], model.input)[0] for i in range(2)]
    grads = tf.stack(grads)
    grads = tf.transpose(grads, perm=[1, 0, 2])

    W = sess.run(model.layers[-1].weights[0].value())
    b = sess.run(model.layers[-1].weights[1].value())

    a_X = deepfool_attack(sess=sess,
                            x=model.input,
                            predictions=model.output,
                            grads=grads,
                            imgs=X_test,
                            n_classes=2,
                            overshoot=0.01,
                            max_iter=1,
                            clip_min=-np.inf,
                            clip_max=np.inf)

    a_y = model.predict(a_X)
    print(a_X.min(), a_X.max())
