import numpy as np

from signxai.methods.grad_cam import calculate_grad_cam_relevancemap, calculate_grad_cam_relevancemap_timeseries
from signxai.methods.guided_backprop import guided_backprop_on_guided_model
from signxai.methods.signed import calculate_sign_mu
from signxai.utils.utils import calculate_explanation_innvestigate


def random_uniform(model_no_softmax, x, **kwargs):
    np.random.seed(1)

    channel_values = []

    uniform_values = np.random.uniform(low=-1, high=1, size=(x.shape[0], x.shape[1]))

    for i in range(x.shape[2]):
        channel_values.append(np.array(uniform_values))

    return np.stack(channel_values, axis=2)


def gradient(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='gradient', **kwargs)


def input_t_gradient(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='input_t_gradient', **kwargs)


def gradient_x_input(model_no_softmax, x, **kwargs):
    g = gradient(model_no_softmax, x, **kwargs)

    return g * x


def gradient_x_sign(model_no_softmax, x, **kwargs):
    g = gradient(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)

    return g * s


def gradient_x_sign_mu(model_no_softmax, x, mu, batchmode=False, **kwargs):
    if batchmode:
        G = []
        S = []
        for xi in x:
            G.append(gradient(model_no_softmax, xi, **kwargs))
            S.append(calculate_sign_mu(xi, mu, **kwargs))
        return np.array(G) * np.array(S)
    else:
        return gradient(model_no_softmax, x, **kwargs) * calculate_sign_mu(x, mu, **kwargs)


def gradient_x_sign_mu_0(model_no_softmax, x, **kwargs):
    return gradient_x_sign_mu(model_no_softmax, x, mu=0, **kwargs)


def gradient_x_sign_mu_0_5(model_no_softmax, x, **kwargs):
    return gradient_x_sign_mu(model_no_softmax, x, mu=0.5, **kwargs)


def gradient_x_sign_mu_neg_0_5(model_no_softmax, x, **kwargs):
    return gradient_x_sign_mu(model_no_softmax, x, mu=-0.5, **kwargs)


def guided_backprop(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='guided_backprop', **kwargs)


def guided_backprop_x_sign(model_no_softmax, x, **kwargs):
    g = guided_backprop(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)

    return g * s


def guided_backprop_x_sign_mu(model_no_softmax, x, mu, batchmode=False, **kwargs):
    if batchmode:
        G = []
        S = []
        for xi in x:
            G.append(guided_backprop(model_no_softmax, xi, **kwargs))
            S.append(calculate_sign_mu(xi, mu, **kwargs))
        return np.array(G) * np.array(S)
    else:
        return guided_backprop(model_no_softmax, x, **kwargs) * calculate_sign_mu(x, mu, **kwargs)


def guided_backprop_x_sign_mu_0(model_no_softmax, x, **kwargs):
    return guided_backprop_x_sign_mu(model_no_softmax, x, mu=0, **kwargs)


def guided_backprop_x_sign_mu_0_5(model_no_softmax, x, **kwargs):
    return guided_backprop_x_sign_mu(model_no_softmax, x, mu=0.5, **kwargs)


def guided_backprop_x_sign_mu_neg_0_5(model_no_softmax, x, **kwargs):
    return guided_backprop_x_sign_mu(model_no_softmax, x, mu=-0.5, **kwargs)

def integrated_gradients(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='integrated_gradients', steps=50, reference_inputs=np.zeros_like(x), **kwargs)


def smoothgrad(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='smoothgrad', augment_by_n=50, noise_scale=0.2, **kwargs)


def smoothgrad_x_sign(model_no_softmax, x, **kwargs):
    g = smoothgrad(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)

    return g * s


def smoothgrad_x_sign_mu(model_no_softmax, x, mu, batchmode=False, **kwargs):
    if batchmode:
        G = []
        S = []
        for xi in x:
            G.append(smoothgrad(model_no_softmax, xi, **kwargs))
            S.append(calculate_sign_mu(xi, mu, **kwargs))
        return np.array(G) * np.array(S)
    else:
        return smoothgrad(model_no_softmax, x, **kwargs) * calculate_sign_mu(x, mu, **kwargs)


def smoothgrad_x_sign_mu_0(model_no_softmax, x, **kwargs):
    return smoothgrad_x_sign_mu(model_no_softmax, x, mu=0, **kwargs)


def smoothgrad_x_sign_mu_0_5(model_no_softmax, x, **kwargs):
    return smoothgrad_x_sign_mu(model_no_softmax, x, mu=0.5, **kwargs)


def smoothgrad_x_sign_mu_neg_0_5(model_no_softmax, x, **kwargs):
    return smoothgrad_x_sign_mu(model_no_softmax, x, mu=-0.5, **kwargs)


def vargrad(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='vargrad', augment_by_n=50, noise_scale=0.2, **kwargs)


def deconvnet(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='deconvnet', **kwargs)


def deconvnet_x_sign(model_no_softmax, x, **kwargs):
    g = deconvnet(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)

    return g * s


def deconvnet_x_sign_mu(model_no_softmax, x, mu, batchmode=False, **kwargs):
    if batchmode:
        G = []
        S = []
        for xi in x:
            G.append(deconvnet(model_no_softmax, xi, **kwargs))
            S.append(calculate_sign_mu(xi, mu, **kwargs))
        return np.array(G) * np.array(S)
    else:
        return deconvnet(model_no_softmax, x, **kwargs) * calculate_sign_mu(x, mu, **kwargs)


def deconvnet_x_sign_mu_0(model_no_softmax, x, **kwargs):
    return deconvnet_x_sign_mu(model_no_softmax, x, mu=0, **kwargs)


def deconvnet_x_sign_mu_0_5(model_no_softmax, x, **kwargs):
    return deconvnet_x_sign_mu(model_no_softmax, x, mu=0.5, **kwargs)


def deconvnet_x_sign_mu_neg_0_5(model_no_softmax, x, **kwargs):
    return deconvnet_x_sign_mu(model_no_softmax, x, mu=-0.5, **kwargs)


def grad_cam(model_no_softmax, x, **kwargs):
    return calculate_grad_cam_relevancemap(np.array([x]), model_no_softmax, resize=True, **kwargs)


def grad_cam_timeseries(model_no_softmax, x, **kwargs):
    return calculate_grad_cam_relevancemap_timeseries(np.array([x]), model_no_softmax, resize=True, **kwargs)


def grad_cam_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return calculate_grad_cam_relevancemap(np.array([x]), model_no_softmax, last_conv_layer_name='block5_conv3', resize=True, **kwargs)


def guided_grad_cam_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    gc = grad_cam_VGG16ILSVRC(model_no_softmax, x, **kwargs)
    gbp = guided_backprop_on_guided_model(model_no_softmax, np.array([x]), layer_name='block5_conv3')

    return gbp * gc


def grad_cam_VGG16MITPL365(model_no_softmax, x, **kwargs):
    return calculate_grad_cam_relevancemap(np.array([x]), model_no_softmax, last_conv_layer_name='relu5_3', resize=True, **kwargs)


def guided_grad_cam_VGG16MITPL365(model_no_softmax, x, **kwargs):
    gc = grad_cam_VGG16MITPL365(model_no_softmax, x, **kwargs)
    gbp = guided_backprop_on_guided_model(model_no_softmax, np.array([x]), layer_name='relu5_3')

    return gbp * gc


def grad_cam_MNISTCNN(model_no_softmax, x, batchmode=False, **kwargs):
    if batchmode:
        H = []
        for xi in x:
            H.append(calculate_grad_cam_relevancemap(np.array([xi]), model_no_softmax, last_conv_layer_name='conv2d_1', resize=True, **kwargs))
        return np.array(H)
    else:
        return calculate_grad_cam_relevancemap(np.array([x]), model_no_softmax, last_conv_layer_name='conv2d_1', resize=True, **kwargs)


def guided_grad_cam_MNISTCNN(model_no_softmax, x, batchmode=False, **kwargs):
    if batchmode:
        gc = grad_cam_MNISTCNN(model_no_softmax, x, batchmode=True, **kwargs)
        gbp = guided_backprop_on_guided_model(model_no_softmax, x, layer_name='conv2d_1')

    else:
        gc = grad_cam_MNISTCNN(model_no_softmax, x, **kwargs)
        gbp = guided_backprop_on_guided_model(model_no_softmax, np.array([x]), layer_name='conv2d_1')

    return gbp * gc


def lrp_z(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.z', **kwargs)


def lrpsign_z(model_no_softmax, x, **kwargs):
    return lrp_z(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def zblrp_z_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_z(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_z(model_no_softmax, x, **kwargs):
    return lrp_z(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_z(model_no_softmax, x, **kwargs):
    return lrp_z(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrp_epsilon_0_001(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=0.001, **kwargs)


def lrpsign_epsilon_0_001(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_001(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def zblrp_epsilon_0_001_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_001(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def lrpz_epsilon_0_001(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_001(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_0_01(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=0.01, **kwargs)


def lrpsign_epsilon_0_01(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_01(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def zblrp_epsilon_0_01_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_01(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_0_01(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_01(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_0_01(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_01(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrpz_epsilon_0_01(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_01(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_0_1(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=0.1, **kwargs)


def lrpsign_epsilon_0_1(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def zblrp_epsilon_0_1_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_0_1(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_0_1(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrpz_epsilon_0_1(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_0_2(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=0.2, **kwargs)


def zblrp_epsilon_0_2_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_2(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def lrpsign_epsilon_0_2(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_2(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_0_2(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_2(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_0_5(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=0.5, **kwargs)


def zblrp_epsilon_0_5_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_5(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def lrpsign_epsilon_0_5(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_5(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_0_5(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_5(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_1(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=1, **kwargs)


def lrpsign_epsilon_1(model_no_softmax, x, **kwargs):
    return lrp_epsilon_1(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def zblrp_epsilon_1_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_1(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_1(model_no_softmax, x, **kwargs):
    return lrp_epsilon_1(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_1(model_no_softmax, x, **kwargs):
    return lrp_epsilon_1(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrpz_epsilon_1(model_no_softmax, x, **kwargs):
    return lrp_epsilon_1(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_5(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=5, **kwargs)


def zblrp_epsilon_5_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_5(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def lrpsign_epsilon_5(model_no_softmax, x, **kwargs):
    return lrp_epsilon_5(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_5(model_no_softmax, x, **kwargs):
    return lrp_epsilon_5(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_10(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=10, **kwargs)


def zblrp_epsilon_10_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_10(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_10(model_no_softmax, x, **kwargs):
    return lrp_epsilon_10(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_10(model_no_softmax, x, **kwargs):
    return lrp_epsilon_10(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrpsign_epsilon_10(model_no_softmax, x, **kwargs):
    return lrp_epsilon_10(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_10(model_no_softmax, x, **kwargs):
    return lrp_epsilon_10(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_20(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=20, **kwargs)


def lrpsign_epsilon_20(model_no_softmax, x, **kwargs):
    return lrp_epsilon_20(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)



def lrpz_epsilon_20(model_no_softmax, x, **kwargs):
    return lrp_epsilon_20(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def zblrp_epsilon_20_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_20(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_20(model_no_softmax, x, **kwargs):
    return lrp_epsilon_20(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_20(model_no_softmax, x, **kwargs):
    return lrp_epsilon_20(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrp_epsilon_50(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=50, **kwargs)


def lrpsign_epsilon_50(model_no_softmax, x, **kwargs):
    return lrp_epsilon_50(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_50(model_no_softmax, x, **kwargs):
    return lrp_epsilon_50(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_75(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=75, **kwargs)


def lrpsign_epsilon_75(model_no_softmax, x, **kwargs):
    return lrp_epsilon_75(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_75(model_no_softmax, x, **kwargs):
    return lrp_epsilon_75(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_100(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=100, **kwargs)


def lrpsign_epsilon_100(model_no_softmax, x, **kwargs):
    return lrp_epsilon_100(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpsign_epsilon_100_mu_0(model_no_softmax, x, **kwargs):
    return lrp_epsilon_100(model_no_softmax, x, input_layer_rule='SIGNmu', mu=0, **kwargs)


def lrpsign_epsilon_100_mu_0_5(model_no_softmax, x, **kwargs):
    return lrp_epsilon_100(model_no_softmax, x, input_layer_rule='SIGNmu', mu=0.5, **kwargs)


def lrpsign_epsilon_100_mu_neg_0_5(model_no_softmax, x, **kwargs):
    return lrp_epsilon_100(model_no_softmax, x, input_layer_rule='SIGNmu', mu=-0.5, **kwargs)


def lrpz_epsilon_100(model_no_softmax, x, **kwargs):
    return lrp_epsilon_100(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def zblrp_epsilon_100_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_100(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_100(model_no_softmax, x, **kwargs):
    return lrp_epsilon_100(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_100(model_no_softmax, x, **kwargs):
    return lrp_epsilon_100(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrp_epsilon_0_1_std_x(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.stdxepsilon', stdfactor=0.1, **kwargs)


def lrpsign_epsilon_0_1_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1_std_x(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_0_1_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1_std_x(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def zblrp_epsilon_0_1_std_x_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1_std_x(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_0_1_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1_std_x(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_0_1_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1_std_x(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrp_epsilon_0_25_std_x(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.stdxepsilon', stdfactor=0.25, **kwargs)


def lrpsign_epsilon_0_25_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_25_std_x(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_0_25_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_25_std_x(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def zblrp_epsilon_0_25_std_x_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_25_std_x(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_0_25_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_25_std_x(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_0_25_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_25_std_x(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrpsign_epsilon_0_25_std_x_mu_0(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_25_std_x(model_no_softmax, x, input_layer_rule='SIGNmu', mu=0, **kwargs)


def lrpsign_epsilon_0_25_std_x_mu_0_5(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_25_std_x(model_no_softmax, x, input_layer_rule='SIGNmu', mu=0.5, **kwargs)


def lrpsign_epsilon_0_25_std_x_mu_neg_0_5(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_25_std_x(model_no_softmax, x, input_layer_rule='SIGNmu', mu=-0.5, **kwargs)


def lrp_epsilon_0_5_std_x(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.stdxepsilon', stdfactor=0.5, **kwargs)


def lrpsign_epsilon_0_5_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_5_std_x(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_0_5_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_5_std_x(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def zblrp_epsilon_0_5_std_x_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_5_std_x(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_0_5_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_5_std_x(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_0_5_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_5_std_x(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrp_epsilon_1_std_x(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.stdxepsilon', stdfactor=1.0, **kwargs)


def lrpsign_epsilon_1_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_1_std_x(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_1_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_1_std_x(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_2_std_x(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.stdxepsilon', stdfactor=2.0, **kwargs)


def lrpsign_epsilon_2_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_2_std_x(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_2_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_2_std_x(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_3_std_x(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.stdxepsilon', stdfactor=3.0, **kwargs)


def lrpsign_epsilon_3_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_3_std_x(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_3_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_3_std_x(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.alpha_1_beta_0', **kwargs)


def lrpsign_alpha_1_beta_0(model_no_softmax, x, **kwargs):
    return lrp_alpha_1_beta_0(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_alpha_1_beta_0(model_no_softmax, x, **kwargs):
    return lrp_alpha_1_beta_0(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def zblrp_alpha_1_beta_0_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_alpha_1_beta_0(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs):
    return lrp_alpha_1_beta_0(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_alpha_1_beta_0(model_no_softmax, x, **kwargs):
    return lrp_alpha_1_beta_0(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrp_sequential_composite_a(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.sequential_composite_a', **kwargs)


def lrpsign_sequential_composite_a(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_a(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_sequential_composite_a(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_a(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def zblrp_sequential_composite_a_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_a(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_sequential_composite_a(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_a(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_sequential_composite_a(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_a(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrp_sequential_composite_b(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.sequential_composite_b', **kwargs)


def lrpsign_sequential_composite_b(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_b(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_sequential_composite_b(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_b(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def zblrp_sequential_composite_b_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_b(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_sequential_composite_b(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_b(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_sequential_composite_b(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_b(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def calculate_relevancemap(m, x, model_no_softmax, **kwargs):
    f = eval(m)
    return f(model_no_softmax, x, **kwargs)


def calculate_relevancemaps(m, X, model_no_softmax, **kwargs):
    Rs = []
    for x in X:
        R = calculate_relevancemap(m, x, model_no_softmax, **kwargs)
        Rs.append(R)

    return np.array(Rs)
