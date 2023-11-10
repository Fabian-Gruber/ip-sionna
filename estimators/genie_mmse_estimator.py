import tensorflow as tf

class genie_mmse_estimator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
    def __call__(self, y, no, C, pilot):
        
        # print('no: ', no)
                                
        # noise_var = no^2 * I. Be careful of the data types!
        noise_var = tf.cast(no * tf.eye(C.shape[-1], batch_shape=[C.shape[0]]), dtype=tf.complex64)
                                        
        # scaled_C = |p|^2 * C
        scaled_C = tf.cast(tf.math.abs(pilot) ** 2, dtype=tf.complex64) * tf.cast(C, dtype=tf.complex64)
        
        # print('check if scaled_C is hermitian', tf.reduce_all(tf.equal(scaled_C, tf.math.conj(tf.transpose(scaled_C, perm=[0, 2, 1])))))
        
        eigenvalues, eigenvectors = tf.linalg.eigh(scaled_C)

        # print('shape of eigenvalues: ', eigenvalues.shape)
                        
        # Compute the inverse of (Lambda * noise_var)^-1
        inverse_lambda_noise_var = tf.linalg.inv(tf.linalg.diag(eigenvalues) + noise_var)

        # Compute the inverse of the sum
        inverse = tf.matmul(tf.matmul(eigenvectors, inverse_lambda_noise_var), tf.transpose(eigenvectors, conjugate=True, perm=[0, 2, 1]))
        
        # print('shape of einsum of inverse_lambda_noise_var: ', tf.einsum('ijk,ilk->ijl', inverse_lambda_noise_var, tf.transpose(eigenvectors, conjugate=True, perm=[0, 1, 2])).shape)
        
        # inverse_2 = tf.einsum('ijk,ikl->ijl', eigenvectors, tf.einsum('ijk,ilk->ijl', inverse_lambda_noise_var, tf.transpose(eigenvectors, conjugate=True, perm=[0, 1, 2])))

        # inverse = tf.linalg.inv(scaled_C + noise_var)
        
        # print('where are inverse and inverse_2 the same: ', tf.reduce_sum(tf.where(tf.abs(inverse - inverse_2) > 1e-2, tf.ones_like(inverse), tf.zeros_like(inverse))))
        
        # print('inverse * (noise + scaled_C): ', tf.matmul(inverse, noise_var + scaled_C)[33, :10, :10])

        # print('eigenvalues: ', eigenvalues[0, :])
        # print('shape of maximum value of eigenvalues: ', tf.math.reduce_max(tf.math.reduce_max(tf.math.real(eigenvalues), axis=1)))
        # print('maximum value of eigenvalues: ', tf.math.reduce_max(tf.math.reduce_max(tf.math.real(eigenvalues), axis=1)))
        # print('minimum value of eigenvalues: ', tf.math.reduce_min(tf.math.reduce_min(tf.math.real(eigenvalues), axis=1)))
        # print('condition number of scaled_C: ', tf.math.reduce_sum(tf.math.reduce_max(tf.math.real(eigenvalues), axis=1) / (tf.math.reduce_min(tf.math.real(eigenvalues), axis=1))) / 500.0)
                
        # scaled_C_2 = conj(p) * C
        scaled_C_2 = tf.math.conj(pilot) * tf.cast(C, dtype=tf.complex64)

        # matrix = scaled_C_2 * inverse
        matrix = tf.matmul(scaled_C_2, inverse)

        # print('first 10 elements of inverse: ', inverse[0, 0, :])

        # print('first 10 elements of scaled_C_2: ', scaled_C_2[0, :, 0])
        # print('first 10 elements of matrix: ', matrix[:5, 10:20, 10:20])
                                                                                                                         
        # h_hat_mmse = (scaled_C_2 * inverse) * y. Be careful of the data types!
        h_hat_mmse = tf.matmul(matrix, tf.transpose(tf.cast(y, dtype=tf.complex64), perm=[0, 2, 1]))   
        # h_hat_mmse_2 = tf.einsum('ijk,ilk->ijl', matrix, y) 

        # print('shape of y: ', y.shape)
        # print('shape of h_hat_mmse: ', h_hat_mmse.shape)

        # print('first 10 elements of y: ', tf.transpose(tf.cast(y, dtype=tf.complex64), perm=[0, 2, 1])[0, :10, 0])
        # print('first 10 elements of h_hat_mmse: ', h_hat_mmse[0, :10, 0])
        # print('first 10 dimensions of matrix: ', matrix[0, :10, :10])

        #compare h_hat_mmse and y
        # print('where are h_hat_mmse and y the same', tf.reduce_sum(tf.where(tf.abs(tf.transpose(h_hat_mmse, perm=[0, 2, 1]) - y) > 1, tf.ones_like(h_hat_mmse), tf.zeros_like(h_hat_mmse))))
        
        
        # print('where are h_hat_mmse and h_hat_mmse_2 the same', tf.reduce_sum(tf.where(tf.abs(h_hat_mmse - h_hat_mmse_2) > 1e-2, tf.ones_like(h_hat_mmse), tf.zeros_like(h_hat_mmse))))
                                                
        return tf.transpose(h_hat_mmse, perm=[0, 2, 1])