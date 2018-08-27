from torch.autograd import Variable
from torch.nn import functional as F


def softmax_scores(compiled_scores, dim=1):
    return F.softmax(Variable(compiled_scores), dim=dim).data


def argmax_scores(compiled_scores, dim=1):
    return compiled_scores.max(dim=dim)[1]


def center_crop_to_reduced_size(tensor, cropped_size_rc, rc_axes=(1, 2)):
    if rc_axes == (1, 2):
        start_coords = (int((tensor.size(1) - cropped_size_rc[0]) / 2),
                        int((tensor.size(2) - cropped_size_rc[1]) / 2))
        cropped_tensor = tensor[:,
                         start_coords[0]:(start_coords[0] + cropped_size_rc[0]),
                         start_coords[1]:(start_coords[1] + cropped_size_rc[1])]
    elif rc_axes == (2, 3):
        assert len(tensor.size()) == 4, NotImplementedError
        start_coords = (int((tensor.size(2) - cropped_size_rc[0]) / 2),
                        int((tensor.size(3) - cropped_size_rc[1]) / 2))
        cropped_tensor = tensor[:, :,
                         start_coords[0]:(start_coords[0] + cropped_size_rc[0]),
                         start_coords[1]:(start_coords[1] + cropped_size_rc[1])]
        assert cropped_tensor.size()[2:] == cropped_size_rc
    else:
        raise NotImplementedError
    return cropped_tensor
