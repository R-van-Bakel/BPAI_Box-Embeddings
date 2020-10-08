import torch
from torch.autograd import Function
import math
from torch.nn import Sigmoid

# class Find_Closest(Function):   #Ruud: Change for per box!
#
#     @staticmethod
#     def forward(ctx, query_box, entity_boxes):
#         ctx.save_for_backward(query_box, entity_boxes)
#         dimensionality = int(query_box.shape[0] / 2)
#
#         closest = []
#
#         for entity in entity_boxes:
#             closest_point = []
#
#             for i in range(dimensionality):
#                 center = entity[i].item()
#                 offset = entity[i + dimensionality].item()
#                 minimum, maximum = sorted((center + offset, center - offset))
#
#                 if query_box[i].item() < minimum:
#                     closest_point.append(minimum)
#                 elif query_box[i].item() > maximum:
#                     closest_point.append(maximum)
#                 else:
#                     closest_point.append(query_box[i].item())
#
#             closest.append(torch.LongTensor(closest_point))
#
#         return torch.stack(closest, 1)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         query_box, entity_boxes = ctx.saved_tensors
#         grad_query_box, grad_entity_boxes = None
#
#         dimensionality = int(query_box.shape[0] / 2)
#
#         closest = []
#
#         for entity in entity_boxes:
#             closest_point = []
#
#             for i in range(dimensionality):
#                 center = entity[i].item()
#                 offset = entity[i + dimensionality].item()
#                 minimum, maximum = sorted((center + offset, center - offset))
#
#                 if query_box[i].item() < minimum:
#                     closest_point.append(minimum)
#                 elif query_box[i].item() > maximum:
#                     closest_point.append(maximum)
#                 else:
#                     closest_point.append(query_box[i].item())
#
#             closest.append(torch.LongTensor(closest_point))
#
#         return grad_query_box, grad_entity_boxes

#################################################################################################

# class Find_Closest(Function):   #Ruud: Change for per box!
#
#     @staticmethod
#     def forward(ctx, query_box, entity_boxes):
#         dimensionality = int(query_box.shape[0] / 2)
#         query_center = query_box[:dimensionality]
#         query_offset = query_box[dimensionality:]
#         entity_center = query_box[:dimensionality]
#         entity_offset = query_box[dimensionality:]
#
#         positive_offset = entity_center + entity_offset
#         negative_offset = entity_center - entity_offset
#
#         entity_minimum = torch.min(positive_offset, negative_offset)
#         entity_maximum = torch.max(positive_offset, negative_offset)
#
#         closest_point = torch.zeros(entity_center.shape, requires_grad=True, dtype=torch.double)
#
#         for i in range(len(query_center)):
#             if query_center[i] < entity_minimum[i]:
#                 closest_point[i] = entity_minimum[i]
#             elif query_center[i] > entity_maximum[i]:
#                 closest_point[i] = entity_maximum[i]
#             else:
#                 closest_point[i] = query_center[i]
#
#         return closest_point
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         pass
#         return


def closest(query_box, entity_box):
    dimensionality = int(query_box.shape[0] / 2)
    query_center = query_box[:dimensionality]
    entity_center = entity_box[:dimensionality]
    entity_offset = entity_box[dimensionality:]

    positive_offset = entity_center + entity_offset
    negative_offset = entity_center - entity_offset

    entity_minimum = torch.min(positive_offset, negative_offset)
    entity_maximum = torch.max(positive_offset, negative_offset)

    if torch.gt(entity_minimum[0], query_center[0]):
        closest_point = torch.stack((entity_minimum[0], entity_minimum[1]))
    elif torch.gt(query_center[0], entity_maximum[0]):
        closest_point = torch.stack((entity_maximum[0], entity_maximum[1]))
    else:
        closest_point = torch.stack((query_center[0], query_center[1]))

    for i in range(2, len(query_center)):
        if torch.gt(entity_minimum[i], query_center[i]):
            closest_point = torch.cat((closest_point, entity_minimum[i].reshape([1])))
        elif torch.gt(query_center[i], entity_maximum[i]):
            closest_point = torch.cat((closest_point, entity_maximum[i].reshape([1])))
        else:
            closest_point = torch.cat((closest_point, query_center[i].reshape([1])))

    return closest_point

# closest = Find_Closest.apply

def return_max(input_box):
    dimensionality = int(input_box.shape[0] / 2)
    return torch.max(input_box[:dimensionality] + input_box[dimensionality:], input_box[:dimensionality] - input_box[dimensionality:])

def return_min(input_box):
    dimensionality = int(input_box.shape[0] / 2)
    return torch.min(input_box[:dimensionality] + input_box[dimensionality:], input_box[:dimensionality] - input_box[dimensionality:])

def distance(entity_box, query_box, alpha=1, device=None):
    dimensionality = int(query_box.shape[0] / 2)
    if device:
        zeros = torch.zeros(dimensionality, dtype=torch.float, requires_grad=False, device=device)
    else:
        zeros = torch.zeros(dimensionality, dtype=torch.float, requires_grad=False)

    closest_point = closest(query_box, entity_box)
    dist_outside = torch.norm(torch.max(closest_point - return_max(query_box), zeros) + torch.max(return_min(query_box) - closest_point, zeros), 1)
    dist_inside = torch.norm(query_box[:dimensionality] - torch.min(return_max(query_box), torch.max(return_min(query_box), closest_point)), 1)
    return dist_outside + torch.mul(dist_inside, alpha)

sigmoid = Sigmoid()

def negative_sampling_loss(query_box, positive_answer_boxes, negative_answer_boxes, gamma, alpha, device=None):
    if device:
        loss = torch.zeros(1, dtype=torch.float, requires_grad=True, device=device)
    else:
        loss = torch.zeros(1, dtype=torch.float, requires_grad=True)

    for box in positive_answer_boxes:
        if len(negative_answer_boxes) > 0:
            loss = torch.sub(loss, (1/len(positive_answer_boxes)) * torch.log(sigmoid(torch.log(gamma) - torch.log(distance(box, query_box, alpha, device) + 0.001))))
    for box in negative_answer_boxes:
        if len(negative_answer_boxes) > 0:
            loss = torch.sub(loss, (1/len(negative_answer_boxes)) * torch.log(sigmoid(torch.log(distance(box, query_box, alpha, device) + 0.001) - torch.log(gamma))))
    return loss
