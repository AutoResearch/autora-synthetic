import copy

import numpy as np
import pandas as pd

from autora_bsr.utils.misc import get_ops_expr
from autora_bsr.utils.node import Node, NodeType
from typing import Dict, List, Optional, Callable, Tuple, Union
from scipy.stats import invgamma, norm
from functools import wraps
from enum import Enum


def check_empty(func: Callable):
    """
    A decorator that, if applied to `func`, checks whether an argument in `func` is an
    un-initialized node (i.e. node.node_type == NodeType.Empty). If so, an error is raised.
    """

    @wraps(func)
    def func_wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, Node):
                if arg.node_type == NodeType.EMPTY:
                    raise TypeError("uninitialized node found in {}".format(func.__name__))
                break
        return func(*args, **kwargs)

    return func_wrapper


@check_empty
def get_height(node: Node) -> int:
    """
    Get the height of a tree starting from `node` as root. The height of a leaf is defined as 0.

    Arguments:
        node: the Node that we hope to calculate `height` for
    Returns:
        height: the height of `node`
    """
    if node.node_type == NodeType.LEAF:
        return 0
    elif node.node_type == NodeType.UNARY:
        return 1 + get_height(node.left)
    else:  # binary node
        return 1 + max(get_height(node.left), get_height(node.right))


@check_empty
def update_depth(node: Node, depth: int):
    node.depth = depth
    if node.node_type == NodeType.UNARY:
        update_depth(node.left, depth + 1)
    elif node.node_type == NodeType.BINARY:
        update_depth(node.left, depth + 1)
        update_depth(node.right, depth + 1)


def get_expression(
        node: Node,
        ops_expr: Dict[str, str] = None,
        feature_names: Optional[List[str]] = None,
) -> str:
    if not ops_expr:
        ops_expr = get_ops_expr()
    if node.node_type == NodeType.LEAF:
        if feature_names:
            return feature_names[node.params["feature"]]
        else:
            return f"x{node.params['feature']}"
    elif node.node_type == NodeType.UNARY:
        # if the expr for an operator is not defined, use placeholder
        # e.g. operator `cosh` -> `cosh(xxx)`
        place_holder = node.op_name + "({})"
        left_expr = get_expression(node.left, ops_expr, feature_names)
        expr_fmt = ops_expr.get(node.op_name, place_holder)
        return expr_fmt.format(left_expr, **node.params)
    elif node.node_type == NodeType.BINARY:
        place_holder = node.op_name + "({})"
        left_expr = get_expression(node.left, ops_expr, feature_names)
        right_expr = get_expression(node.right, ops_expr, feature_names)
        expr_fmt = ops_expr.get(node.op_name, place_holder)
        return expr_fmt.format(left_expr, right_expr, **node.params)
    else:  # empty node
        return "(empty node)"


@check_empty
def get_all_nodes(node: Node) -> List[Node]:
    """
    Get all the nodes below (and including) the given `node` via pre-order traversal

    Return:
        a list with all the nodes below (and including) the given `node`
    """
    nodes = [node]
    if node.node_type == NodeType.UNARY:
        nodes.extend(get_all_nodes(node.left))
    elif node.node_type == NodeType.BINARY:
        nodes.extend(get_all_nodes(node.left))
        nodes.extend(get_all_nodes(node.right))
    return nodes


@check_empty
def get_num_lt_nodes(node: Node):
    if node.node_type == NodeType.LEAF:
        return 0
    else:
        base = 1 if node.op_name == "ln" else 0
        if node.node_type == NodeType.UNARY:
            return base + get_num_lt_nodes(node.left)
        else:
            return base + get_num_lt_nodes(node.left) + get_num_lt_nodes(node.right)


def calc_tree_ll(
        node: Node,
        ops_priors: Dict[str, Dict],
        n_feature: int = 1,
        **hyper_params: Dict
):
    struct_ll = 0  # log likelihood of tree structure S = (T,M)
    params_ll = 0  # log likelihood of linear params
    depth = node.depth
    beta = hyper_params.get("beta", -1)
    sigma_a, sigma_b = hyper_params.get("sigma_a", 1), hyper_params.get("sigma_b", 1)
    op_weight = ops_priors[node.op_name]["weight"]

    # contribution of hyperparameter sigma_theta
    if not depth:  # root node
        struct_ll += np.log(invgamma.pdf(sigma_a, 1))
        struct_ll += np.log(invgamma.pdf(sigma_b, 1))

    # contribution of splitting the node or becoming leaf node
    if node.node_type == NodeType.LEAF:
        # contribution of choosing terminal
        struct_ll += np.log(
            1 - 1 / np.power((1 + depth), -beta)
        )
        # contribution of feature selection
        struct_ll -= np.log(n_feature)
        return struct_ll, params_ll
    elif node.node_type == NodeType.UNARY:  # unitary operator
        # contribution of child nodes are added since the log likelihood is additive
        # if we assume the parameters are independent.
        struct_ll_left, params_ll_left = calc_tree_ll(node.left, ops_priors, n_feature, **hyper_params)
        struct_ll += struct_ll_left
        params_ll += params_ll_left
        # contribution of parameters of linear nodes
        # make sure the below parameter ll calculation is extendable
        if node.op_name == "ln":
            params_ll -= np.power((node.params["a"] - 1), 2) / (2 * sigma_a)
            params_ll -= np.power(node.params["b"], 2) / (2 * sigma_b)
            params_ll -= 0.5 * np.log(4 * np.pi ** 2 * sigma_a * sigma_b)
    else:  # binary operator
        struct_ll_left, params_ll_left = calc_tree_ll(node.right, ops_priors, n_feature, **hyper_params)
        struct_ll_right, params_ll_right = calc_tree_ll(node.right, ops_priors, n_feature, **hyper_params)
        struct_ll += struct_ll_left + struct_ll_right
        params_ll += params_ll_left + params_ll_right

    # for unary & binary nodes, additionally consider the contribution of splitting
    if not depth:  # root node
        struct_ll += np.log(op_weight)
    else:
        struct_ll += np.log((1 + depth)) * beta + np.log(op_weight)

    return struct_ll, params_ll


def calc_y_ll(y: np.ndarray, outputs: Union[np.ndarray, pd.DataFrame], sigma_y: float):
    """
    Calculate the log likelihood f(y|S,Theta,x) where (S,Theta) is represented by the node
    prior is y ~ N(output,sigma) and output is the matrix of outputs corresponding to different roots

    Returns:
        log_sum: the data log likelihood
    """
    outputs = copy.deepcopy(outputs)
    scale = np.max(np.abs(outputs))
    outputs = outputs / scale
    epsilon = np.eye(outputs.shape[1]) * 1e-6
    beta = np.linalg.inv(np.matmul(outputs.transpose(), outputs) + epsilon)
    beta = np.matmul(beta, np.matmul(outputs.transpose(), y))
    # perform the linear combination
    output = np.matmul(outputs, beta)
    # calculate the squared error
    error = np.sum(np.square(y - output[:, 0]))

    log_sum = error
    var = 2 * sigma_y * sigma_y
    log_sum = -log_sum / var
    log_sum -= 0.5 * len(y) * np.log(np.pi * var)
    return log_sum


def stay(ln_nodes: List[Node], **hyper_params: Dict):
    """
    ACTION 1: Stay represents the action of doing nothing but to update the parameters for `ln`
    operators.

    Arguments:
        ln_nodes: the list of nodes with `ln` operator
        hyper_params: hyperparameters for re-initialization
    """
    for ln_node in ln_nodes:
        ln_node.init_param(**hyper_params)


def grow(
        node: Node,
        ops_name_lst: List[str],
        ops_weight_lst: List[float],
        ops_priors: Dict[str, Dict],
        n_feature: int = 1,
        **hyper_params: Dict
):
    """
    ACTION 2: Grow represents the action of growing a subtree from a given `node`

    Arguments:
        node: the tree node from where the subtree starts to grow
        ops_name_lst: list of operation names
        ops_weight_lst: list of operation prior weights
        ops_priors: the dictionary of operation prior properties
        n_feature: the number of features in input data
        hyper_params: hyperparameters for re-initialization
    """
    depth = node.depth
    p = 1 / np.power((1 + depth), -hyper_params.get("beta", -1))

    if depth > 0 and p < np.random.uniform(0, 1, 1):  # create leaf node
        node.setup(feature=np.random.randint(0, n_feature, 1))
    else:
        ops_name = np.random.choice(ops_name_lst, p=ops_weight_lst)
        ops_prior = ops_priors[ops_name]
        node.setup(ops_name, ops_prior, **hyper_params)

        # recursively set up downstream nodes
        grow(node.left, ops_name_lst, ops_weight_lst, ops_priors, n_feature, **hyper_params)
        if node.node_type == NodeType.BINARY:
            grow(node.right, ops_name_lst, ops_weight_lst, ops_priors, n_feature, **hyper_params)


@check_empty
def prune(node: Node, n_feature: int = 1):
    """
    ACTION 3: Prune a non-terminal node into a terminal node and assign it a feature

    Arguments:
        node: the tree node to be pruned
        n_feature: the number of features in input data
    """
    node.setup(feature=np.random.randint(0, n_feature, 1))


@check_empty
def de_transform(node: Node) -> Tuple[Node, Optional[Node]]:
    """
    ACTION 4: De-transform deletes the current `node` and replaces it with children
    according to the following rule: if the `node` is unary, simply replace with its
    child; if `node` is binary and root, choose any children that's not leaf; if `node`
    is binary and not root, pick any children.

    Arguments:
        node: the tree node that gets de-transformed

    Returns:
        first node is the replaced node when `node` has been de-transformed
        second node is the discarded node
    """
    if node.node_type == NodeType.UNARY:
        return node.left, None
    r = np.random.random()
    # picked node is root
    if not node.depth:
        if node.left.node_type == NodeType.LEAF:
            return node.right, node.left
        elif node.right.node_type == NodeType.LEAF:
            return node.left, node.right
        else:
            return node.left, node.right if r < 0.5 else node.right, node.left
    elif r < 0.5:
        return node.left, node.right
    else:
        return node.right, node.left


@check_empty
def transform(
        node: Node,
        ops_name_lst: List[str],
        ops_weight_lst: List[float],
        ops_priors: Dict[str, Dict],
        n_feature: int = 1,
        **hyper_params: Dict
) -> Node:
    """
    ACTION 5: Transform inserts a middle node between the picked `node` and its
    parent. Assign an operation to this middle node using the priors. If the middle
    node is binary, `grow` its right child. The left child of the middle node is
    set to `node` and its parent becomes `node.parent`.

    Arguments:
        node: the tree node that gets transformed
        ops_name_lst: list of operation names
        ops_weight_lst: list of operation prior weights
        ops_priors: the dictionary of operation prior properties
        n_feature: the number of features in input data
        hyper_params: hyperparameters for re-initialization
    Return:
        the middle node that gets inserted
    """
    parent = node.parent

    insert_node = Node(depth=node.depth, parent=parent)
    insert_op = np.random.choice(ops_name_lst, 1, ops_weight_lst)[0]
    insert_node.setup(insert_op, ops_priors[insert_op], **hyper_params)

    if parent:
        is_left = node is parent.left
        if is_left:
            parent.left = insert_node
        else:
            parent.right = insert_node

    # set the left child as `node` and grow the right child if needed (binary case)
    insert_node.left = node
    node.parent = insert_node
    if insert_node.node_type == NodeType.BINARY:
        grow(insert_node.right, ops_name_lst, ops_weight_lst, ops_priors, n_feature, **hyper_params)

    # make sure the depth property is updated correctly
    update_depth(node, node.depth + 1)
    return insert_node


@check_empty
def reassign_op(
        node: Node,
        ops_name_lst: List[str],
        ops_weight_lst: List[float],
        ops_priors: Dict[str, Dict],
        n_feature: int = 1,
        **hyper_params: Dict
):
    """
    ACTION 6: Re-assign action uniformly picks a non-terminal node, and assign a new operator.
    If the node changes from unary to binary, its original child is taken as the left child,
    and we grow a new subtree as right child. If the node changes from binary to unary, we
    preserve the left subtree (this is to make the transition reversible).

    Arguments:
        node: the tree node that gets re-assigned an operator
        ops_name_lst: list of operation names
        ops_weight_lst: list of operation prior weights
        ops_priors: the dictionary of operation prior properties
        n_feature: the number of features in input data
        hyper_params: hyperparameters for re-initialization
    """
    # make sure `node` is non-terminal
    old_type = node.node_type
    assert old_type != NodeType.LEAF

    # store the original children and re-setup the `node`
    old_left, old_right = node.left, node.right
    new_op = np.random.choice(ops_name_lst, 1, ops_weight_lst)[0]
    node.setup(new_op, ops_priors[new_op], **hyper_params)

    new_type = node.node_type

    node.left = old_left
    if old_type == new_type:  # binary -> binary & unary -> unary
        node.right = old_right
    elif new_type == NodeType.BINARY:  # unary -> binary
        grow(node.right, ops_name_lst, ops_weight_lst, ops_priors, n_feature, **hyper_params)
    else:
        node.right = None


@check_empty
def reassign_feat(node: Node, n_feature: int = 1):
    """
    ACTION 7: Re-assign feature randomly picks a feature and assign it to `node`.

    Arguments:
        node: the tree node that gets re-assigned a feature
        n_feature: the number of features in input data
    """
    # make sure we have a leaf node
    assert node.node_type == NodeType.LEAF
    node.setup(feature=np.random.randint(0, n_feature, 1))


class Action(Enum):
    STAY = 0
    GROW = 1
    PRUNE = 2
    DE_TRANSFORM = 3
    TRANSFORM = 4
    REASSIGN_OP = 5
    REASSIGN_FEAT = 6

    @classmethod
    def rand_action(
            cls,
            lt_num: int,
            term_num: int,
            de_trans_num: int
    ) -> Tuple[int, float]:
        """
        Draw a random action for MCMC algorithm to take a step

        Arguments:
            lt_num: the number of linear (`lt`) nodes in the tree
            term_num: the number of terminal nodes in the tree
            de_trans_num: the number of de-trans qualified nodes in the tree (see `propose` for details)
        Returns:
            action: the MCMC action to perform
            weights: the probabilities for each action
        """
        # from the BSR paper
        weights = []
        weights[0] = 0.25 * lt_num / (lt_num + 3)  # p_stay
        weights[1] = (1 - weights[0]) * min(1, 4 / (term_num + 2)) / 3  # p_grow
        weights[2] = (1 - weights[0]) / 3 - weights[1]  # p_prune
        weights[3] = (1 - weights[0]) * (1 / 3) * de_trans_num / (3 + de_trans_num)  # p_detrans
        weights[4] = (1 - weights[0]) / 3 - weights[3]  # p_trans
        weights[5] = (1 - weights[0]) / 6  # p_reassign_op
        weights[6] = 1 - sum(weights)  # p_reassign_feat
        assert weights[6] >= 0

        action = np.random.choice(np.arange(7), p=weights)
        return action, weights


def _get_tree_classified_nodes(root: Node, get_count=False) \
        -> Union[Tuple[List[Node], List[Node], List[Node], List[Node]],
                 Tuple[int, int, int, int]]:
    """
    calculate the classified lists of nodes from a tree

    Argument:
        root: the root node where the calculation starts from
        get_count: if `True`, the method returns the `len` of each list instead of the
            list itself.
    Returns:
        term_nodes: the list of terminal nodes (or the count of this list, same below)
        nterm_nodes: the list of non-terminal nodes
        lt_nodes: the list of nodes with linear operator
        de_trans_nodes: the list of nodes that can be de-transformed
    """
    term_nodes: List[Node] = []
    nterm_nodes: List[Node] = []
    lt_nodes: List[Node] = []
    de_trans_nodes: List[Node] = []
    for node in get_all_nodes(root):
        if node.node_type == NodeType.LEAF:
            term_nodes.append(node)
        else:
            nterm_nodes.append(node)
            # rules for deciding whether a non-terminal node is de-transformable
            # 1. node is not root OR 2. children are not both terminal nodes
            if node.depth or (node.left or node.right):
                de_trans_nodes.append(node)
        if node.op_name == "ln":
            lt_nodes.append(node)

    if get_count:
        return len(term_nodes), len(nterm_nodes), len(lt_nodes), len(de_trans_nodes)
    else:
        return term_nodes, nterm_nodes, lt_nodes, de_trans_nodes


@check_empty
def prop(
        node: Node,
        ops_name_lst: List[str],
        ops_weight_lst: List[float],
        ops_priors: Dict[str, Dict],
        n_feature: int = 1,
        **hyper_params
):
    """
    Propose a new tree from an existing tree with root `node`

    Return:
        new_node: the new node after some action is applied
        expand_node: whether the node has been expanded
        shrink_node: whether the node has been shrunk
        q: quantities for calculating acceptance prob
        q_inv: quantities for calculating acceptance prob
    """
    # PART 1: collect necessary information
    new_node = copy.deepcopy(node)
    term_nodes, nterm_nodes, lt_nodes, de_trans_nodes = _get_tree_classified_nodes(new_node)

    # PART 2: sample random action and perform the action
    # this step also calculates q and q_inv, quantities necessary for calculating
    # the acceptance probability in MCMC algorithm
    action, probs = Action.rand_action(len(lt_nodes), len(term_nodes), len(de_trans_nodes))
    # flags indicating potential dimensionality change (expand or shrink) in node
    expand_node, shrink_node = False, False

    # ACTION 1: STAY
    # q and q_inv simply equal the probability of choosing this action
    if action == Action.STAY:
        q = probs[Action.STAY]
        q_inv = probs[Action.STAY]
        stay(new_node, **hyper_params)
    # ACTION 2: GROW
    # q and q_inv simply equal the probability if the grown node is a leaf node
    # otherwise, we calculate new information of the `new_node` after the action is applied
    elif action == Action.GROW:
        i = np.random.randint(0, len(term_nodes), 1)[0]
        grown_node: Node = term_nodes[i]
        grow(grown_node, ops_name_lst, ops_weight_lst, ops_priors, n_feature, **hyper_params)
        if grown_node.node_type == NodeType.LEAF:
            q = q_inv = 1
        else:
            tree_ll, param_ll = calc_tree_ll(grown_node, ops_priors, n_feature, **hyper_params)
            # calculate q
            q = probs[Action.GROW] * np.exp(tree_ll) / len(term_nodes)
            # calculate q_inv by using updated information of `new_node`
            new_term_count, new_nterm_count, new_lt_count, _ = _get_tree_classified_nodes(new_node, get_count=True)
            new_prob = (
                    (1 - 0.25 * new_lt_count / (new_lt_count + 3))
                    * (1 - min(1, 4 / (len(new_nterm_count) + 2)))
                    / 3
            )
            q_inv = new_prob / max(1, new_nterm_count - 1)  # except the root
            if new_lt_count > len(lt_nodes):
                expand_node = True

    elif action == Action.PRUNE:
        i = np.random.randint(0, len(nterm_nodes), 1)[0]
        pruned_node: Node = nterm_nodes[i]
        prune(pruned_node, n_feature)
        tree_ll, param_ll = calc_tree_ll(pruned_node, ops_priors, n_feature, **hyper_params)

        new_term_count, new_nterm_count, new_lt_count, _ = _get_tree_classified_nodes(new_node, get_count=True)
        # pruning any tree with `ln` operator will result in shrinkage
        if new_lt_count < len(lt_nodes):
            shrink_node = True

        # calculate q
        q = probs[Action.PRUNE] / ((len(nterm_nodes) - 1) * n_feature)
        pg = 1 - 0.25 * new_lt_count / (new_lt_count + 3) * 0.75 * min(
            1, 4 / (new_nterm_count + 2)
        )
        # calculate q_inv
        q_inv = pg * np.exp(tree_ll) / new_term_count

    elif action == Action.DE_TRANSFORM:
        num_de_trans = len(de_trans_nodes)
        i = np.random.randint(0, num_de_trans, 1)[0]
        de_trans_node: Node = de_trans_nodes[i]
        replaced_node, discarded_node = de_transform(de_trans_node)
        par_node = de_trans_node.parent

        q = probs[Action.DE_TRANSFORM] / num_de_trans
        if not par_node and de_trans_node.left and de_trans_node.right \
                and de_trans_node.left.node_type != NodeType.LEAF \
                and de_trans_node.right.node_type != NodeType.LEAF:
            q = q / 2
        elif de_trans_node.node_type == NodeType.BINARY:
            q = q / 2

        if not par_node:  # de-transformed the root
            new_node = replaced_node
            new_node.parent = None
            update_depth(new_node, 0)
        elif par_node.left is de_trans_node:
            par_node.left = replaced_node
            replaced_node.parent = par_node
            update_depth(replaced_node, par_node.depth + 1)
        else:
            par_node.right = replaced_node
            replaced_node.parent = par_node
            update_depth(replaced_node, par_node.depth + 1)

        new_term_count, new_nterm_count, new_lt_count, new_det_count = \
            _get_tree_classified_nodes(new_node, get_count=True)

        if new_lt_count < len(lt_nodes):
            shrink_node = True

        new_prob = 0.25 * new_lt_count / (new_lt_count + 3)
        # calculate q_inv
        new_pdetr = (1 - new_prob) * (1 / 3) * new_det_count / (new_det_count + 3)
        new_ptr = (1 - new_prob) / 3 - new_pdetr
        q_inv = new_ptr * ops_priors[de_trans_node.op_name]["weight"] / (new_term_count + new_nterm_count)
        if discarded_node:
            tree_ll, _ = calc_tree_ll(discarded_node, ops_priors, n_feature, **hyper_params)
            q_inv = q_inv * np.exp(tree_ll)

    elif action == Action.TRANSFORM:
        all_nodes = get_all_nodes(new_node)
        i = np.random.randint(0, len(all_nodes), 1)[0]
        trans_node: Node = all_nodes[i]
        inserted_node: Node = \
            transform(trans_node, ops_name_lst, ops_weight_lst, ops_priors, n_feature, **hyper_params)

        if inserted_node.right:
            ll_right, _ = calc_tree_ll(inserted_node.right, ops_priors, n_feature, **hyper_params)
        else:
            ll_right = 0
        # calculate q
        q = probs[Action.TRANSFORM] * ops_priors[inserted_node.op_name]["weight"] * np.exp(ll_right) / len(all_nodes)

        new_term_count, new_nterm_count, new_lt_count, new_det_count = \
            _get_tree_classified_nodes(new_node, get_count=True)
        if new_lt_count > len(lt_nodes):
            expand_node = True

        new_prob = 0.25 * new_lt_count / (new_lt_count + 3)
        # calculate q_inv
        new_pdetr = (1 - new_prob) * (1 / 3) * new_det_count / (new_det_count + 3)
        q_inv = new_pdetr / new_det_count
        if inserted_node.node_type == NodeType.BINARY and \
                inserted_node.left.node_type != NodeType.LEAF and inserted_node.right.node_type != NodeType.LEAF:
            q_inv = q_inv / 2

    elif action == Action.REASSIGN_OP:
        i = np.random.randint(0, len(nterm_nodes), 1)[0]
        reassign_node: Node = nterm_nodes[i]
        old_right = reassign_node.right
        old_op_name, old_type = reassign_node.op_name, reassign_node.node_type
        reassign_op(reassign_node, ops_name_lst, ops_weight_lst, ops_priors, n_feature, **hyper_params)
        new_type = reassign_node.node_type
        _, new_nterm_count, new_lt_count, _ = _get_tree_classified_nodes(new_node, get_count=True)

        if old_type == new_type:  # binary -> binary & unary -> unary
            q = ops_priors[reassign_node.op_name]["weight"]
            q_inv = ops_priors[old_op_name]["weight"]
        else:
            op_weight = ops_priors[reassign_node.op_name]["weight"]
            if old_type == NodeType.UNARY:  # unary -> binary
                tree_ll, _ = calc_tree_ll(reassign_node.right, ops_priors, n_feature, **hyper_params)
                q = probs[Action.REASSIGN_OP] * np.exp(tree_ll) * op_weight / len(nterm_nodes)
                ll_factor = 1
            else:  # binary -> unary
                tree_ll, _ = calc_tree_ll(old_right, ops_priors, n_feature, **hyper_params)
                q = probs[Action.REASSIGN_OP] * op_weight / len(nterm_nodes)
                ll_factor = tree_ll
            # calculate q_inv
            new_prob = new_lt_count / (4 * (new_lt_count + 3))
            q_inv = (
                0.125
                * (1 - new_prob)
                * ll_factor
                * ops_priors[old_op_name]["weight"]
                / new_nterm_count
            )
        if new_lt_count > len(lt_nodes):
            expand_node = True
        elif new_lt_count < len(lt_nodes):
            shrink_node = True

    else:  # reassign feature
        i = np.random.randint(0, len(term_nodes), 1)[0]
        reassign_node = term_nodes[i]
        reassign_feat(reassign_node, n_feature)
        q = q_inv = 1

    return new_node, expand_node, shrink_node, q, q_inv


def calc_aux_ll(node: Node, **hyper_params):
    sigma_a, sigma_b = hyper_params["sigma_a"], hyper_params["sigma_b"]
    log_aux = np.log(invgamma.pdf(sigma_a, 1)) + np.log(invgamma.pdf(sigma_b, 1))

    all_nodes = get_all_nodes(node)
    lt_count = 0
    for i in range(all_nodes):
        if all_nodes[i].op_name == "ln":
            lt_count += 1
            a, b = all_nodes[i].params["a"], all_nodes[i].params["b"]
            log_aux += np.log(norm.pdf(a, 1, np.sqrt(sigma_a)))
            log_aux += np.log(norm.pdf(b, 0, np.sqrt(sigma_b)))

    return log_aux, lt_count


def prop_new(
        roots: List[Node],
        index: int,
        sigma_y: float,
        beta: float,
        sigma_a: float,
        sigma_b: float,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        ops_name_lst: List[str],
        ops_weight_lst: List[float],
        ops_priors: Dict[str, Dict],
):
    # the hyper-param for linear combination, i.e. for `sigma_y`
    sig = 4
    K = len(roots)
    root = roots[index]
    use_aux_ll = True

    # sample new sigma_a and sigma_b
    new_sigma_a = invgamma.rvs(1)
    new_sigma_b = invgamma.rvs(1)

    hyper_params = {"sigma_a": sigma_a, "sigma_b": sigma_b, "beta": beta}
    new_hyper_params = {"sigma_a": new_sigma_a, "sigma_b": new_sigma_b, "beta": beta}
    # propose a new tree `node`
    new_root, expand_node, shrink_node, q, q_inv = \
        prop(root, ops_name_lst, ops_weight_lst, ops_priors, X.shape[1], **new_hyper_params)

    n_feature = X.shape[0]
    new_outputs = np.zeros((len(y), K))
    old_outputs = np.zeros((len(y), K))

    for i in np.arange(K):
        tmp_old = root.evaluate(X)
        old_outputs[:, i] = tmp_old
        if i == index:
            new_outputs[:, i] = new_root.evaluate(X)
        else:
            new_outputs[:, i] = tmp_old

    if np.linalg.matrix_rank(new_outputs) < K:  # rejection due to insufficient rank
        return False, root, sigma_y, sigma_a, sigma_b

    y_ll_old = calc_y_ll(y, old_outputs, sigma_y)
    # a magic number here as the parameter for generating new sigma_y
    new_sigma_y = invgamma.rvs(sig)
    y_ll_new = calc_y_ll(y, new_outputs, new_sigma_y)

    log_y_ratio = y_ll_new - y_ll_old
    # contribution of f(Theta, S)
    if shrink_node or expand_node:
        struct_ll_old = sum(calc_tree_ll(root, ops_priors, n_feature, **hyper_params))
        struct_ll_new = sum(calc_tree_ll(new_root, ops_priors, n_feature, **hyper_params))
        log_struct_ratio = struct_ll_new - struct_ll_old
    else:
        log_struct_ratio = calc_tree_ll(new_root, ops_priors, n_feature, **hyper_params)[0] - \
            calc_tree_ll(root, ops_priors, n_feature, **hyper_params)

    # contribution of proposal Q and Qinv
    log_q_ratio = np.log(max(1e-5, q_inv / q))

    log_r = log_y_ratio + log_struct_ratio + log_q_ratio + \
        np.log(invgamma.pdf(new_sigma_y, sig)) - np.log(invgamma.pdf(sigma_y, sig))

    if use_aux_ll and (expand_node or shrink_node):
        old_aux_ll, old_lt_count = calc_aux_ll(root, **hyper_params)
        new_aux_ll, _ = calc_aux_ll(new_root, **new_hyper_params)
        log_r += old_aux_ll - new_aux_ll
        # log for the Jacobian
        log_r += np.log(max(1e-5, 1 / np.power(2, 2 * old_lt_count)))

    alpha = min(log_r, 0)
    test = np.random.uniform(0, 1, 0)[0]
    if np.log(test) >= alpha:  # no accept
        return False, root, sigma_y, sigma_a, sigma_b
    else:  # accept
        return True, new_root, new_sigma_y, new_sigma_a, new_sigma_b
