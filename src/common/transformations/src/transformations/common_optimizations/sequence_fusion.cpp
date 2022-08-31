// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/sequence_fusion.hpp"

#include <memory>
#include <openvino/opsets/opset9.hpp>

#include "itt.hpp"
#include "ngraph_ops/augru_cell.hpp"
#include "ngraph_ops/augru_sequence.hpp"

using namespace std;
using namespace ov;
using namespace opset9;
using namespace pass::pattern;
using namespace ov::op::util;

namespace {
bool is_equal_consts(const shared_ptr<Node>& l, const shared_ptr<Node>& r) {
    auto l_const = dynamic_pointer_cast<Constant>(l);
    auto r_const = dynamic_pointer_cast<Constant>(r);
    if (l_const && r_const) {
        std::cout << "XXXX COnstants" << std::endl;

        bool is_equal =
            l_const->get_element_type() == r_const->get_element_type() && l_const->get_shape() == r_const->get_shape();
        if (!is_equal) {
            std::cout << "" << std::endl;
        }
        return is_equal;
    }
    std::cout << "XXXX name " << l->get_friendly_name();
    std::cout << "XXXX name " << r->get_friendly_name();
    return false;
}

bool check_WRB(const shared_ptr<RNNCellBase>& cell_1, const shared_ptr<RNNCellBase>& cell_2) {
    int64_t idx_W = 2, idx_R = 3, idx_B = 4;
    auto lstm_cell_1 = dynamic_pointer_cast<LSTMCell>(cell_1);
    auto lstm_cell_2 = dynamic_pointer_cast<LSTMCell>(cell_2);

    // 2nd input is Cell State
    if (lstm_cell_1 && lstm_cell_2) {
        idx_B++;
        idx_R++;
        idx_W++;
    }

    auto lW = cell_1->input_value(idx_W).get_node_shared_ptr();
    auto lR = cell_1->input_value(idx_R).get_node_shared_ptr();
    auto lB = cell_1->input_value(idx_B).get_node_shared_ptr();
    auto rW = cell_2->input_value(idx_W).get_node_shared_ptr();
    auto rR = cell_2->input_value(idx_R).get_node_shared_ptr();
    auto rB = cell_2->input_value(idx_B).get_node_shared_ptr();
    bool is_equal = lW.get() == rW.get() && lR.get() == rR.get() && lB.get() == rB.get();
    if (!is_equal) {
        is_equal = is_equal_consts(lW, rW) && is_equal_consts(lR, rR) && is_equal_consts(lB, rB);
    }
    return is_equal;
}

bool is_equal_cells(const shared_ptr<RNNCellBase>& cell_1, const shared_ptr<RNNCellBase>& cell_2) {
    bool is_equal =
        cell_1->get_type_name() == cell_2->get_type_name() && cell_1->get_hidden_size() == cell_2->get_hidden_size() &&
        cell_1->get_activations() == cell_2->get_activations() &&
        cell_1->get_activations_alpha() == cell_2->get_activations_alpha() &&
        cell_1->get_activations_beta() == cell_2->get_activations_beta() && cell_1->get_clip() == cell_2->get_clip();
    is_equal &= check_WRB(cell_1, cell_2);
    auto gru_cell_1 = dynamic_pointer_cast<GRUCell>(cell_1);
    auto gru_cell_2 = dynamic_pointer_cast<GRUCell>(cell_2);
    if (gru_cell_1 && gru_cell_2) {
        is_equal &= gru_cell_1->get_linear_before_reset() == gru_cell_2->get_linear_before_reset();
    }
    return is_equal;
}

shared_ptr<RNNCellBase> find_cell_chain(ov::pass::NodeRegister& cp_from,
                                        ov::pass::NodeRegister& cp_to,
                                        const shared_ptr<RNNCellBase>& current_cell,
                                        OutputVector& x_to_concat,
                                        OutputVector& attention_to_concat,
                                        map<int, set<ov::Input<Node>>>& h_inputs_to_redirect,
                                        map<int, set<ov::Input<Node>>>& c_inputs_to_redirect,
                                        int& cells_cnt,
                                        const shared_ptr<Node>& axis_1) {
    cells_cnt++;

    shared_ptr<RNNCellBase> current = current_cell;
    while (true) {
        cp_from.add(current);
        // check the source node of HiddenState input
        auto prev = current->input_value(1).get_node_shared_ptr();
        if (auto prev_cell = dynamic_pointer_cast<RNNCellBase>(prev)) {
            if (!is_equal_cells(prev_cell, current)) {
                std::cout << "XXXXXXXX Cells are not equal" << std::endl;
                std::cout << "Cells cnt : " << cells_cnt << std::endl;
                break;
            }

            auto in_X = current->input(0);
            x_to_concat.push_back(cp_to.make<Unsqueeze>(in_X.get_source_output(), axis_1));

            // collect inputs (target_inputs) connected to H output of prev_node except H input of the current node
            auto in_H = current->input(1);
            for (const auto& input : prev_cell->get_output_target_inputs(0)) {
                if (input != in_H) {
                    h_inputs_to_redirect[cells_cnt].insert(input);
                }
            }

            if (auto lstm = dynamic_pointer_cast<LSTMCell>(current)) {
                auto in_C = current->input(2);
                // collect inputs (target_inputs) connected to C output of prev_node except C input of the current node
                for (const auto& input : prev_cell->get_output_target_inputs(1)) {
                    if (input != in_C) {
                        c_inputs_to_redirect[cells_cnt].insert(input);
                    }
                }
            }

            if (auto augru = dynamic_pointer_cast<ov::op::internal::AUGRUCell>(prev_cell)) {
                attention_to_concat.push_back(cp_to.make<Unsqueeze>(augru->input_value(5), axis_1));
            }

            current = prev_cell;
            cells_cnt++;
        } else {
            auto in_X = current->input(0);
            x_to_concat.push_back(cp_to.make<Unsqueeze>(in_X.get_source_output(), axis_1));
            if (auto augru = dynamic_pointer_cast<ov::op::internal::AUGRUCell>(current)) {
                attention_to_concat.push_back(cp_to.make<Unsqueeze>(augru->input_value(5), axis_1));
            }
            break;
        }
    }
    reverse(x_to_concat.begin(), x_to_concat.end());
    reverse(attention_to_concat.begin(), attention_to_concat.end());
    // the first cell in the chain
    return current;
}

bool create_sequence(ov::pass::NodeRegister& cp_to,
                     const shared_ptr<RNNCellBase>& first_cell,
                     const shared_ptr<RNNCellBase>& last_cell,
                     const OutputVector& x_to_concat,
                     const OutputVector& attention_to_concat,
                     const map<int, set<ov::Input<Node>>>& h_inputs_to_redirect,
                     const map<int, set<ov::Input<Node>>>& c_inputs_to_redirect,
                     int cells_cnt,
                     const shared_ptr<Node>& axis_0,
                     const shared_ptr<Node>& axis_1) {
    int64_t idx_W = 2, idx_R = 3, idx_B = 4;
    auto lstm_cell_1 = dynamic_pointer_cast<LSTMCell>(last_cell);
    // 2nd input is Cell State
    if (lstm_cell_1) {
        idx_B++;
        idx_R++;
        idx_W++;
    }

    const auto X_in = cp_to.make<Concat>(x_to_concat, 1);
    const auto Ht_in = cp_to.make<Unsqueeze>(first_cell->input_value(1), axis_1);
    const auto W_in = cp_to.make<Unsqueeze>(first_cell->input_value(idx_W), axis_0);
    const auto R_in = cp_to.make<Unsqueeze>(first_cell->input_value(idx_R), axis_0);
    const auto B_in = cp_to.make<Unsqueeze>(first_cell->input_value(idx_B), axis_0);

    const auto& shape_node = cp_to.add(ngraph::op::util::make_try_fold<ShapeOf>(first_cell->input_value(0)));
    const auto& zero = cp_to.make<Constant>(ov::element::i64, Shape{1}, 0);
    const auto& batch_dimension = cp_to.add(ngraph::op::util::make_try_fold<Gather>(shape_node, zero, axis_0));
    auto seq_lengths_scalar = cp_to.make<Constant>(element::i64, Shape{}, cells_cnt);
    auto sequence_lengths_in =
        cp_to.add(ngraph::op::util::make_try_fold<Broadcast>(seq_lengths_scalar, batch_dimension));

    shared_ptr<Node> sequence;
    OutputVector outputs(1);
    if (dynamic_pointer_cast<LSTMCell>(first_cell)) {
        cout << "XXXXXXXX LSTMSequence pattern detected" << endl;
        const auto Ct_in = cp_to.make<Unsqueeze>(first_cell->input_value(2), axis_1);
        sequence = cp_to.make<LSTMSequence>(X_in,
                                            Ht_in,
                                            Ct_in,
                                            sequence_lengths_in,
                                            W_in,
                                            R_in,
                                            B_in,
                                            first_cell->get_hidden_size(),
                                            ov::op::RecurrentSequenceDirection::FORWARD,
                                            first_cell->get_activations_alpha(),
                                            first_cell->get_activations_beta(),
                                            first_cell->get_activations(),
                                            first_cell->get_clip());
        if (!c_inputs_to_redirect.empty()) {
            // if intermediate C outputs are used in the network,
            // then we cannot fuse Cells to Sequence.
            // Sequence doesn't provide access to these C outputs.
            return false;
        }
        outputs.resize(2);
        outputs[1] = cp_to.make<Squeeze>(sequence->output(2), axis_1);
    } else if (auto gru_cell = dynamic_pointer_cast<GRUCell>(first_cell)) {
        cout << "XXXXXXXX GRUSequence pattern detected" << endl;
        sequence = cp_to.make<GRUSequence>(X_in,
                                           Ht_in,
                                           sequence_lengths_in,
                                           W_in,
                                           R_in,
                                           B_in,
                                           first_cell->get_hidden_size(),
                                           ov::op::RecurrentSequenceDirection::FORWARD,
                                           first_cell->get_activations(),
                                           first_cell->get_activations_alpha(),
                                           first_cell->get_activations_beta(),
                                           first_cell->get_clip(),
                                           gru_cell->get_linear_before_reset());
    } else if (dynamic_pointer_cast<RNNCell>(first_cell)) {
        cout << "XXXXXXXX RNNSequence pattern detected" << endl;
        sequence = cp_to.make<RNNSequence>(X_in,
                                           Ht_in,
                                           sequence_lengths_in,
                                           W_in,
                                           R_in,
                                           B_in,
                                           first_cell->get_hidden_size(),
                                           ov::op::RecurrentSequenceDirection::FORWARD,
                                           first_cell->get_activations(),
                                           first_cell->get_activations_alpha(),
                                           first_cell->get_activations_beta(),
                                           first_cell->get_clip());
    } else if (dynamic_pointer_cast<ov::op::internal::AUGRUCell>(first_cell)) {
        cout << "XXXXXXXX AUGRUSequence pattern detected" << endl;
        const auto A_in = cp_to.make<Concat>(attention_to_concat, 1);
        sequence = cp_to.make<ov::op::internal::AUGRUSequence>(X_in,
                                                               Ht_in,
                                                               sequence_lengths_in,
                                                               W_in,
                                                               R_in,
                                                               B_in,
                                                               A_in,
                                                               first_cell->get_hidden_size());
    } else {
        // cell is not supported;
        return false;
    }

    outputs[0] = cp_to.make<Squeeze>(sequence->output(1), axis_1);
    replace_outputs_update_names(last_cell->outputs(), outputs);
    cout << "XXXXXXXX Sequence pattern replaced" << endl;

    if (!h_inputs_to_redirect.empty()) {
        auto squeeze_Y = cp_to.make<Squeeze>(sequence->output(0), axis_1);
        auto split = cp_to.make<Split>(squeeze_Y, axis_1, cells_cnt);

        for (const auto& it : h_inputs_to_redirect) {
            for (const auto& in : it.second) {
                auto squeeze = cp_to.make<Squeeze>(split->output(cells_cnt - it.first - 1), axis_1);
                in.replace_source_output(squeeze);
            }
        }
    }
    return true;
}
}  // namespace

ov::pass::SequenceFusion::SequenceFusion() {
    MATCHER_SCOPE(SequenceFusion);

    auto cell = wrap_type<RNNCellBase>();
    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        cout << "XXXXXXXX Sequance Transformation" << endl;
        NodeRegister copy_from;
        NodeRegister copy_to;
        auto cell = m.get_match_root();
        shared_ptr<RNNCellBase> current_cell = dynamic_pointer_cast<RNNCellBase>(cell);
        if (!current_cell) {
            std::cout << "XXXX Exit 1" << std::endl;
            return false;
        }

        // check that this is the last Cell in the chain, e.g.
        // GRUCell -> GRUCell (the last cell) -> OtherNode
        // GRUCell (hidden_size = 128) -> GRUCell (hs = 128, the last) -> GRUCell (hs = 64)
        for (const auto& target : cell->get_output_target_inputs(0)) {
            auto cell_1 = dynamic_pointer_cast<RNNCellBase>(target.get_node()->shared_from_this());
            if (cell_1 && is_equal_cells(cell_1, current_cell)) {
                std::cout << "XXXX Exit 2" << std::endl;
                return false;
            }
        }

        int cells_cnt = 0;
        OutputVector x_to_concat;
        OutputVector attention_to_concat;
        map<int, set<ov::Input<Node>>> h_inputs_to_redirect;
        map<int, set<ov::Input<Node>>> c_inputs_to_redirect;
        auto axis_0 = copy_to.make<Constant>(element::i64, Shape{}, 0);
        auto axis_1 = copy_to.make<Constant>(element::i64, Shape{}, 1);

        // detect chain (Cell->Cell->Cell->..)
        auto first_cell = find_cell_chain(copy_from,
                                          copy_to,
                                          current_cell,
                                          x_to_concat,
                                          attention_to_concat,
                                          h_inputs_to_redirect,
                                          c_inputs_to_redirect,
                                          cells_cnt,
                                          axis_1);

        // no reasons to create sequence if the single cell detected.
        // investigate optimal cnt of cells
        int optimal_cnt_of_cells = 2;
        if (cells_cnt < optimal_cnt_of_cells) {
            std::cout << "XXXX Exit 3" << std::endl;
            return false;
        }

        auto res = create_sequence(copy_to,
                                   first_cell,
                                   current_cell,
                                   x_to_concat,
                                   attention_to_concat,
                                   h_inputs_to_redirect,
                                   c_inputs_to_redirect,
                                   cells_cnt,
                                   axis_0,
                                   axis_1);
        if (!res) {
            std::cout << "XXXX Exit 4" << std::endl;
            return false;
        }
        // copy_runtime_info(copy_from.get(), copy_to.get());
        return true;
    };

    auto m = make_shared<Matcher>(cell, matcher_name);
    this->register_matcher(m, callback);
}