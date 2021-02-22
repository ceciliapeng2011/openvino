//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <functional>
#include <onnx/onnx_pb.h>
#include <stack>

#include "ngraph/check.hpp"
#include "onnx_import/editor/detail/subgraph_extraction.hpp"

using namespace ngraph::onnx_import;

namespace
{
    void validate_node_index(const ONNX_NAMESPACE::GraphProto& graph, const int node_idx)
    {
        NGRAPH_CHECK(
            node_idx >= 0 && node_idx < graph.node_size(),
            "The specified node index is out of range of nodes in the original model(idx: ",
            std::to_string(node_idx),
            "; nodes count in the model: ",
            std::to_string(graph.node_size()),
            ")");
    }

    template <typename T>
    std::function<bool(const T&)> name_equals(const std::string& name)
    {
        return [&name](const T& onnx_object) -> bool { return onnx_object.name() == name; };
    }

    const auto is_equal_to =
        +[](const std::string& other) { return [&](const std::string& s) { return s == other; }; };

    template <typename Container>
    bool already_exists(const Container& items, const std::string& name)
    {
        using std::begin;
        using std::end;
        return std::any_of(
            begin(items), end(items), name_equals<typename Container::value_type>(name));
    }

    bool is_graph_input(const ONNX_NAMESPACE::GraphProto& graph, const std::string& name)
    {
        return already_exists(graph.input(), name);
    }

    bool is_graph_initializer(const ONNX_NAMESPACE::GraphProto& graph, const std::string& name)
    {
        return already_exists(graph.initializer(), name);
    }

    int find_source_node_idx(const ONNX_NAMESPACE::GraphProto& graph,
                             const int current_node_idx,
                             const std::string& input_name)
    {
        for (int i = current_node_idx - 1; i >= 0; --i)
        {
            const auto& outputs = graph.node(i).output();
            const auto output_found =
                std::any_of(std::begin(outputs), std::end(outputs), is_equal_to(input_name));

            if (output_found)
            {
                return i;
            }
        }

        throw ngraph::ngraph_error{"Source node not found in the graph for node: " +
                                   std::to_string(current_node_idx) + " and input name: " +
                                   input_name};
    }

    /// \brief Looks up a descriptor for a given tensor name. This descriptor contains inferred
    ///        shape information which is required to create new inputs and outputs in the graph.
    const ONNX_NAMESPACE::ValueInfoProto&
        find_tensor_descriptor(const ONNX_NAMESPACE::GraphProto& graph,
                               const std::string& tensor_name)
    {
        const auto it = std::find_if(std::begin(graph.value_info()),
                                     std::end(graph.value_info()),
                                     name_equals<ONNX_NAMESPACE::ValueInfoProto>(tensor_name));

        NGRAPH_CHECK(it != std::end(graph.value_info()),
                     "Could not find a tensor descriptor for tensor '",
                     tensor_name,
                     "'. It's not possible to add a new input to the graph without the type and "
                     "shape information of the intermediate tensor.");

        return *it;
    }

    void replace_initializer_with_new_input(ONNX_NAMESPACE::GraphProto& graph,
                                            const InputEdge& edge)
    {
        const auto it = std::find_if(std::begin(graph.initializer()),
                                     std::end(graph.initializer()),
                                     name_equals<ONNX_NAMESPACE::TensorProto>(edge.m_tensor_name));

        NGRAPH_CHECK(it != std::end(graph.initializer()),
                     "Could not find an initializer in the graph: '",
                     edge.m_tensor_name);

        if (!already_exists(graph.input(), edge.m_tensor_name))
        {
            const auto& initializer = *it;
            auto& new_input = *(graph.add_input());

            auto& new_input_tensor_type = *(new_input.mutable_type()->mutable_tensor_type());
            new_input_tensor_type.set_elem_type(initializer.data_type());

            auto& new_input_shape = *(new_input_tensor_type.mutable_shape());
            for (const auto initializer_dim : initializer.dims())
            {
                auto& new_dim = *(new_input_shape.add_dim());
                new_dim.set_dim_value(initializer_dim);
            }

            *(new_input.mutable_name()) = edge.m_tensor_name;
        }

        graph.mutable_initializer()->erase(it);
    }

    std::pair<bool, InputEdge> append_new_graph_input(ONNX_NAMESPACE::GraphProto& graph,
                                                      const InputEdge& edge)
    {
        if (already_exists(graph.input(), edge.m_tensor_name) &&
            !is_graph_initializer(graph, edge.m_tensor_name))
        {
            // no need to append a new input if an edge points to an existing one in the model
            return {false, edge};
        }

        auto& target_node = *(graph.mutable_node(edge.m_node_idx));
        auto& node_inputs = *(target_node.mutable_input());
        auto target_input =
            std::find(std::begin(node_inputs), std::end(node_inputs), edge.m_tensor_name);

        NGRAPH_CHECK(target_input != std::end(node_inputs),
                     "Input '",
                     edge.m_tensor_name,
                     "' not found in the inputs of node ",
                     edge.m_node_idx,
                     ". Cannot append a new graph input to this node.");

        const std::string new_input_name = target_node.output(0) + ":" + edge.m_tensor_name;

        if (is_graph_initializer(graph, edge.m_tensor_name))
        {
            replace_initializer_with_new_input(graph, edge);
            return {false, edge};
        }
        else
        {
            auto& new_input = *(graph.add_input());
            // copy the intermediate tensor properties to the newly created input
            new_input.MergeFrom(find_tensor_descriptor(graph, edge.m_tensor_name));
            *(new_input.mutable_name()) = new_input_name;
            // attach the new graph input to the target node's input
            *target_input = new_input_name;
            return {true, InputEdge{edge.m_node_idx, new_input_name}};
        }
    }

    /// \brief Replaces a node or initializer (consumed by multiple nodes) with a new input
    int replace_source_with_new_input(ONNX_NAMESPACE::GraphProto& graph, const InputEdge& edge)
    {
        if (already_exists(graph.input(), edge.m_tensor_name) &&
            !is_graph_initializer(graph, edge.m_tensor_name))
        {
            // happens when a user specifies multiple input edges pointing to the same tensor name
            return -1;
        }

        if (is_graph_initializer(graph, edge.m_tensor_name))
        {
            replace_initializer_with_new_input(graph, edge);
        }
        else
        {
            auto& new_input = *(graph.add_input());
            // copy the intermediate tensor properties to the newly created input
            new_input.MergeFrom(find_tensor_descriptor(graph, edge.m_tensor_name));

            const auto source_node_idx =
                find_source_node_idx(graph, edge.m_node_idx, edge.m_tensor_name);
            auto& source_node = *(graph.mutable_node(source_node_idx));
            auto& node_outputs = *source_node.mutable_output();
            auto target_output =
                std::find(std::begin(node_outputs), std::end(node_outputs), edge.m_tensor_name);

            NGRAPH_CHECK(target_output != std::end(node_outputs),
                         "Output '",
                         edge.m_tensor_name,
                         "' not found in the outputs of node ",
                         source_node_idx,
                         ". Cannot remove the output from this node.");

            // stop produsing tensor "edge.m_tensor_name" by the source node of the processed edge
            *target_output = "";

            return source_node_idx;
        }

        return -1;
    }

    void append_new_graph_output(ONNX_NAMESPACE::GraphProto& graph, const OutputEdge& edge)
    {
        if (already_exists(graph.output(), edge.m_tensor_name))
        {
            return;
        }

        auto& target_node = *(graph.mutable_node(edge.m_node_idx));
        const auto& node_outputs = target_node.output();
        const auto target_output =
            std::find(std::begin(node_outputs), std::end(node_outputs), edge.m_tensor_name);

        NGRAPH_CHECK(target_output != std::end(node_outputs),
                     "Output '",
                     edge.m_tensor_name,
                     "' not found in the outputs of node ",
                     edge.m_node_idx,
                     ". Cannot append a new graph output to this node.");

        auto& new_output = *(graph.add_output());
        // copy the intermediate tensor's properties to the newly created
        new_output.MergeFrom(find_tensor_descriptor(graph, edge.m_tensor_name));
        *(new_output.mutable_name()) = edge.m_tensor_name;
    }

    /// \brief Removes all items from a container except the ones whose names are in items_to_keep
    ///        It's intended to work with ONNX graph inputs, outputs and initializers only.
    template <typename Container>
    void discard_by_name(Container& all_items, const std::set<std::string>& items_to_keep)
    {
        static_assert(
            std::is_same<typename Container::value_type, ONNX_NAMESPACE::ValueInfoProto>::value ||
                std::is_same<typename Container::value_type, ONNX_NAMESPACE::TensorProto>::value,
            "Unsupported value type of the container");

        // The tested item can be discarded if its name is not found in the items_to_keep set
        const auto can_be_discarded = [&items_to_keep](const typename Container::value_type& item) {
            return items_to_keep.count(item.name()) == 0;
        };

        using std::begin;
        using std::end;

        // move the elements-to-discard to the end of the container
        const auto new_end = std::remove_if(begin(all_items), end(all_items), can_be_discarded);
        // erase all of the discarded elements past the new end of the container
        all_items.erase(new_end, end(all_items));
    }

    /// \brief Removes all nodes from a container keeping the ones whose index is in nodes_to_keep
    template <typename Container>
    void discard_nodes(Container& all_nodes, const std::set<int>& nodes_to_keep)
    {
        static_assert(
            std::is_same<typename Container::value_type, ONNX_NAMESPACE::NodeProto>::value,
            "Unsupported value type of the container");

        int idx = 0;
        const auto discard_node = [&idx, &nodes_to_keep](const typename Container::value_type&) {
            return nodes_to_keep.count(idx++) == 0;
        };

        using std::begin;
        using std::end;

        const auto new_end = std::remove_if(begin(all_nodes), end(all_nodes), discard_node);
        all_nodes.erase(new_end, end(all_nodes));
    }
} // namespace

/* -----------------------------------------------------------------------------------------------*/




SubgraphExtractor::SubgraphExtractor(ONNX_NAMESPACE::GraphProto& graph)
    : m_onnx_graph(graph)
{
    for (int i = 0; i < graph.node_size(); ++i)
    {
        for (const auto& node_input : graph.node(i).input())
        {
            m_node_inputs.insert({i, node_input});
            m_tensor_consumers[node_input] += 1;
        }
    }
}

void SubgraphExtractor::add_new_inputs(const std::vector<InputEdge>& new_inputs)
{
    for (const auto& edge_to_replace : new_inputs)
    {
        validate_node_index(m_onnx_graph, edge_to_replace.m_node_idx);

        if (m_tensor_consumers[edge_to_replace.m_tensor_name] > 1)
        {
            int idx = replace_source_with_new_input(m_onnx_graph, edge_to_replace);
            if (idx != -1)
            {
                // if a node was replaced with an input, remove input edges from a helper multimap
                // for this node because it won't end up in the target subgraph
                m_node_inputs.erase(idx);
            }
        }
        else
        {
            const auto& new_edge = append_new_graph_input(m_onnx_graph, edge_to_replace);
            if (new_edge.first)
            {
                replace_input_edge(edge_to_replace, new_edge.second);
            }
        }
    }
}

void SubgraphExtractor::add_new_outputs(const std::vector<OutputEdge>& new_outputs)
{
    for (const auto& new_output : new_outputs)
    {
        validate_node_index(m_onnx_graph, new_output.m_node_idx);

        append_new_graph_output(m_onnx_graph, new_output);
    }
}

void SubgraphExtractor::replace_input_edge(const InputEdge& old_edge, const InputEdge& new_edge)
{
    const auto node_inputs = m_node_inputs.equal_range(old_edge.m_node_idx);
    auto old_input_name = node_inputs.first;

    while (old_input_name->second != old_edge.m_tensor_name && old_input_name != node_inputs.second)
    {
        ++old_input_name;
    }

    m_node_inputs.erase(old_input_name);
    m_node_inputs.insert({new_edge.m_node_idx, new_edge.m_tensor_name});
}

void SubgraphExtractor::extract_subgraph(std::vector<OutputEdge> subgraph_outputs)
{
    if (subgraph_outputs.empty())
    {
        subgraph_outputs = all_output_edges();
    }

    SubgraphComponents subgraph;

    for (const auto& output_edge : subgraph_outputs)
    {
        subgraph += discover_output_contributors(output_edge, subgraph);
    }

    extract_subgraph_from_onnx_model(subgraph);
}

SubgraphExtractor::SubgraphComponents SubgraphExtractor::discover_output_contributors(
    const OutputEdge& output_edge, const SubgraphComponents& already_collected) const
{
    const auto already_visited = [&already_collected](const int node_index) {
        return already_collected.nodes.count(node_index) > 0;
    };

    SubgraphComponents output_contributors;
    output_contributors.outputs.insert(output_edge.m_tensor_name);

    // reverse DFS graph traversal
    std::stack<int> nodes_to_visit;
    nodes_to_visit.push(output_edge.m_node_idx);

    while (!nodes_to_visit.empty())
    {
        const auto n = nodes_to_visit.top();
        nodes_to_visit.pop();

        if (already_visited(n))
        {
            continue;
        }

        output_contributors.nodes.insert(n);

        // check if the visitor reached any of the graph inputs
        // and/or keep looking for more contributors further up in the graph
        const auto n_inputs = m_node_inputs.equal_range(n);
        for (auto input_name = n_inputs.first; input_name != n_inputs.second; ++input_name)
        {
            if (is_graph_input(m_onnx_graph, input_name->second))
            {
                output_contributors.inputs.insert(input_name->second);
                // when an initializer has a matching graph input
                if (is_graph_initializer(m_onnx_graph, input_name->second))
                {
                    output_contributors.initializers.insert(input_name->second);
                }
            }
            else if (is_graph_initializer(m_onnx_graph, input_name->second))
            {
                // when an initializer doesn't have a corresponding input
                output_contributors.initializers.insert(input_name->second);
            }
            else
            {
                nodes_to_visit.push(find_source_node_idx(m_onnx_graph, n, input_name->second));
            }
        }
    }

    return output_contributors;
}

void SubgraphExtractor::extract_subgraph_from_onnx_model(const SubgraphComponents& subgraph)
{
    discard_by_name(*(m_onnx_graph.mutable_input()), subgraph.inputs);
    discard_by_name(*(m_onnx_graph.mutable_initializer()), subgraph.initializers);
    discard_by_name(*(m_onnx_graph.mutable_output()), subgraph.outputs);
    discard_nodes(*(m_onnx_graph.mutable_node()), subgraph.nodes);
}

std::vector<OutputEdge> SubgraphExtractor::all_output_edges() const
{
    std::vector<OutputEdge> all_outputs;

    for (const auto& graph_output : m_onnx_graph.output())
    {
        all_outputs.emplace_back(
            find_source_node_idx(m_onnx_graph, m_onnx_graph.node_size(), graph_output.name()),
            graph_output.name());
    }

    return all_outputs;
}
