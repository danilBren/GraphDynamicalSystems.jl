import AutoHashEquals: @auto_hash_equals
import DynamicalSystemsBase: current_parameters, get_state, set_state!
import SciMLBase

using AbstractTrees: Leaves, PostOrderDFS
using DynamicalSystemsBase: ArbitrarySteppable, initial_state
using Graphs: AbstractGraph, SimpleDiGraph, add_edge!, add_vertex!, ne
using HerbConstraints: DomainRuleNode, Forbidden, Ordered, Unique, VarNode, addconstraint!
using HerbCore: AbstractGrammar, RuleNode, get_rule
using HerbGrammar: @csgrammar, add_rule!, rulenode2expr
using HerbSearch: rand
using MLStyle: @match
using MetaGraphsNext: MetaGraph, add_edge!, edge_labels, inneighbor_labels, labels, nv
using StaticArrays: MVector, SVector

const base_qn_grammar = @csgrammar begin
    Val = Val + Val
    Val = Val - Val
    Val = Val / Val
    Val = Val * Val
    Val = min(Val, Val)
    Val = max(Val, Val)
    Val = ceil(Val)
    Val = floor(Val)
end

const default_qn_constants = [0, 1, 2]

"""
    $(TYPEDSIGNATURES)

Builds a grammar based on the base QN grammar adding `entity_names` and `constants`
to the grammar.

The following constraints are currently included

1. removing symmetry due to commutativity of `+`/`*`/`min`/`max`
2. forbidding same arguments of two argument functions
3. forbidding constant arguments to 2-argument functions
4. forbidding constant arguments to 1-argument functions
5. using each of the entities only once per function
6. forbidding adding or subtracting zero
7. forbidding multiplication and division by 1 or 0
8. forcing the first operator inside `ceil` and `floor` to be `÷`
9. forbidding `max(□, X)` and `min(□, X)` where X is either the max or min
constant in the grammar.

"""
function build_qn_grammar(
    entity_names,
    constants = default_qn_constants;
    unique_constr = true,
)
    g = deepcopy(base_qn_grammar)

    for e in entity_names
        add_rule!(g, :(Val = $e))
    end

    for c in constants
        add_rule!(g, :(Val = $c))
    end

    add_rule!(g, :(Start = Val))

    # +, *, min, max, are all commutative
    domain = BitVector(zeros(length(g.rules)))
    @. domain[[1, 4:6...]] = true
    template_tree = DomainRuleNode(domain, [VarNode(:a), VarNode(:b)])
    order = [:a, :b]

    addconstraint!(g, Ordered(deepcopy(template_tree), order))

    # Forbid same arguments for 2-argument functions
    domain = BitVector(zeros(length(g.rules)))
    @. domain[length(g.childtypes)==2] = true
    template_tree = DomainRuleNode(domain, [VarNode(:a), VarNode(:a)])

    addconstraint!(g, Forbidden(deepcopy(template_tree)))

    # Forbid constant arguments for 2-argument functions
    domain = falses(length(g.rules))
    @. domain[length(g.childtypes)==2] = true
    consts_domain = falses(length(g.rules))
    consts_domain[findall(x -> x isa Int, g.rules)] .= true
    consts_domain_rn = DomainRuleNode(consts_domain)
    template_tree = DomainRuleNode(domain, [consts_domain_rn, consts_domain_rn])

    addconstraint!(g, Forbidden(deepcopy(template_tree)))

    # Forbid constant arguments for 1-argument functions
    domain = falses(length(g.rules))
    @. domain[[7, 8]] = true
    consts_domain = falses(length(g.rules))
    consts_domain[findall(x -> x isa Int, g.rules)] .= true
    consts_domain_rn = DomainRuleNode(consts_domain)
    template_tree = DomainRuleNode(domain, [consts_domain_rn])

    addconstraint!(g, Forbidden(deepcopy(template_tree)))

    n_original_rules = length(base_qn_grammar.rules)

    # Only use each of the entities once per function
    n_consts = length(constants)
    entities = (n_original_rules+1):(length(g.rules)-n_consts)

    if unique_constr
        addconstraint!.((g,), Unique.(entities))
    end

    # Forbid □ + 0, □ - 0
    plus_or_minus = falses(length(g.rules))
    plus_or_minus[[1, 2]] .= true
    zero_rule = findfirst(==(0), g.rules)
    if !isnothing(zero_rule)
        template_tree = DomainRuleNode(plus_or_minus, [VarNode(:a), RuleNode(zero_rule)])

        addconstraint!(g, Forbidden(deepcopy(template_tree)))

        # Both orderings, but only for plus. Allow 0 - □
        plus_or_minus[2] = false
        template_tree = DomainRuleNode(plus_or_minus, [RuleNode(zero_rule), VarNode(:a)])
        addconstraint!(g, Forbidden(deepcopy(template_tree)))
    end

    # Forbid □ * 1, □ / 1, □ * 0, □ / 0
    mult_or_div = falses(length(g.rules))
    mult_or_div[[3, 4]] .= true
    one_zero_domain = falses(length(g.rules))
    one_zero_domain[findfirst(==(1), g.rules)] = true
    if !isnothing(findfirst(==(0), g.rules))
        one_zero_domain[findfirst(==(0), g.rules)] = true
    end

    template_tree =
        DomainRuleNode(mult_or_div, [VarNode(:a), DomainRuleNode(one_zero_domain)])

    addconstraint!(g, Forbidden(deepcopy(template_tree)))

    # Forbid ceil(X) and floor(X) unless X = □ ÷ □
    ceil_or_floor = BitVector(zeros(length(g.rules)))
    ceil_or_floor[[7, 8]] .= true
    all_except_div = trues(length(g.rules))
    all_except_div[3] = false
    template_tree = DomainRuleNode(ceil_or_floor, [DomainRuleNode(all_except_div)])

    addconstraint!(g, Forbidden(deepcopy(template_tree)))

    # Forbid max(□, X) and min(□, X) where X is either the largest or smallest constant in the grammar
    min_max_rules = falses(length(g.rules))
    min_max_rules[[5, 6]] .= true
    (min_const, max_const) = extrema(filter(x -> isa(x, Int), g.rules))
    extrema_domain = falses(length(g.rules))
    extrema_domain[findall(x -> x == min_const || x == max_const, g.rules)] .= true
    rule_extrema_consts = DomainRuleNode(extrema_domain)
    template_tree = DomainRuleNode(min_max_rules, [VarNode(:a), rule_extrema_consts])

    addconstraint!(g, Forbidden(deepcopy(template_tree)))

    return g
end

"""
    $TYPEDSIGNATURES

Construct a default target function for an entity in a QN from a list of
`activators` and `inhibitors`.

Follows the definition given in Eq. 3 of ["Qualitative networks: a symbolic
approach to analyze biological signaling
networks"](https://doi.org/10.1186/1752-0509-1-4).

## Examples

Say we have a component `X` and it has an lower bound on its state value of 0,
an upper bound of 4, activators `A`, `B`, `C`, and inhibitors `D`, `E`, `F`,
then the following example constructs an expression for its default target
function.

```jldoctest
julia> default_target_function(0, 4, [:A, :B, :C], [:D, :E, :F])
:(max(0, (A + B + C) / 3 - (D + E + F) / 3))
```

"""
function default_target_function(
    lower_bound::Integer,
    upper_bound::Integer,
    activators::AbstractVector = [],
    inhibitors::AbstractVector = [],
)
    sum_only_or_nothing = x -> if length(x) == 0
        nothing
    elseif length(x) == 1
        :($(only(x)))
    elseif length(x) > 1
        :($(Expr(:call, :+, x...)) / $(length(x)))
    end

    expr_activators = sum_only_or_nothing(activators)
    expr_inhibitors = sum_only_or_nothing(inhibitors)

    if isnothing(expr_activators) && isnothing(expr_inhibitors)
        error("Constructing a default target function for a QN with no \
              activators or inhibitors.")
    elseif isnothing(expr_activators) # no activators, special case mentioned in paper
        return :($upper_bound - $expr_inhibitors)
    elseif isnothing(expr_inhibitors)
        return :($expr_activators)
    else
        return :(max($lower_bound, $expr_activators - $expr_inhibitors))
    end
end

abstract type EntityLabel end

struct EntityId <: EntityLabel
    id::Int
end
id(e::EntityId) = e.id

struct EntityName <: EntityLabel
    name::Symbol
end
name(e::EntityName) = e.name

Base.:(==)(A::EntityName, B::EntityName) = A.name == B.name
Base.isless(A::EntityName, B::EntityName) = A.name < B.name
convert(::Type{EntityName}, S::Symbol) = EntityName(S)
convert(::Type{Symbol}, EN::EntityName) = EN.name
Base.show(io::IO, E::EntityName) = print(io, E.name)
@auto_hash_equals struct EntityIdName{S} <: EntityLabel
    id::Int
    name::S
end
function EntityIdName(s::Symbol)
    en_str = string(s)
    name_id_str_split = rsplit(en_str, "_"; limit = 2)
    if length(name_id_str_split) != 2
        error("""Failed to convert the Symbol $s to an EntityIdName. \
              Expecting an EntityName with a name in the form of "Name_00".""")
    end
    (name_str, id_str) = name_id_str_split

    id_val = tryparse(Int, id_str)
    if isnothing(id_val)
        error("""Entity name ($s) contained an underscore but the \
              content after the underscore ($id_str) could not be parsed as \
              an integer to convert it to an ID.""")
    end

    return EntityIdName(id_val, string(name_str))
end
EntityIdName{String}(s::Symbol) = EntityIdName(s)
id(e::EntityIdName) = e.id
name(e::EntityIdName) = e.name
combined_name(e::EntityIdName) = Symbol("$(name(e))_$(id(e))")

mutable struct Entity{I<:EntityLabel,D}
    label::I
    target_function::Any
    domain::UnitRange{D}
end
Entity(name::Symbol, args...) = Entity(EntityName(name), args...)
Entity(id::Int, args...) = Entity(EntityId(id), args...)
Entity((id, name), args...) = Entity(EntityIdName(id, name), args...)

Base.show(io::Core.IO, e::Entity) = print(io, "(|$(e.label), $(e.target_function), $(e.domain)|)")
Base.isless(E1::Entity, E2::Entity) = E1.label < E2.label

label(e::Entity) = e.label
id(e::Entity) = id(label(e))
name(e::Entity) = name(label(e))
target_function(e::Entity) = e.target_function
domain(e::Entity) = e.domain
range_from(e::Entity) = first(domain(e))
range_to(e::Entity) = last(domain(e))

function get_used_entities(fn, entities_in_model::Vector{<:Entity{<:EntityIdName}})
    filter(in(combined_name.(label.(entities_in_model))), collect(Leaves(fn)))
end

function get_used_entities(fn, entities_in_model)
    filter(in(name.(entities_in_model)), collect(Leaves(fn)))
end

"""
    $(TYPEDSIGNATURES)
"""
function update_functions_to_interaction_graph(
    entities_in_model::AbstractVector{<:E};
    schedule = Synchronous,
) where {EntityLabelType,E<:Entity{EntityLabelType}}
    graph = MetaGraph(
        SimpleDiGraph();
        label_type = EntityLabelType,
        vertex_data_type = E,
        graph_data = schedule,
    )
    if !allunique(label.(entities_in_model))
        val_counts = Dict()
        for e in entities_in_model
            val_counts[name(e)] = append!(get(val_counts, name(e), []), [e])
        end
        duplicates = [v for v in values(val_counts) if length(v) > 1]
        error("""The QN implementation only supports models with unique \
              entity name/id combinations.

              Duplicates:

              $duplicates""")
    end

    for entity in entities_in_model
        graph[label(entity)] = entity
    end

    for dst in entities_in_model
        input_entities = get_used_entities(target_function(dst), entities_in_model)
        for src in EntityLabelType.(input_entities)
            dst_label = label(dst)
            l = collect(labels(graph))
            if !(src ∈ l && dst_label ∈ l)
                error(
                    """Could not add edge from $src to $(dst_label). The vertex labels in the graph are currently $(collect(labels(graph))).""",
                )
            end
            add_edge!(graph, src, dst_label)
        end
    end

    return graph
end

"""
    $(TYPEDSIGNATURES)
"""
function sample_qualitative_network(
    entities::AbstractVector{Entity},
    domains::AbstractVector{UnitRange{Int}},
    max_eq_depth::Int;
    schedule = Synchronous,
)
    g = build_qn_grammar(entities, default_qn_constants)
    update_fns = Union{Expr,Integer,Symbol}[
        rulenode2expr(rand(RuleNode, g, :Val, max_eq_depth), g) for _ in entities
    ]

    qn = QualitativeNetwork(Entity.(entities, update_fns, domains); schedule = schedule)

    return qn
end

sample_qualitative_network(N::Int, args...; kwargs...) =
    sample_qualitative_network(Entity.(Symbol.(('A':'Z')[1:N])), args...; kwargs...)

"""
    $(TYPEDEF)

A qualitative network model as described in ["Qualitative networks: a symbolic approach to
analyze biological signaling networks"](https://doi.org/10.1186/1752-0509-1-4).

This implementation encompasses both the synchronous and asynchonous cases. In the paper, it
is assumed that the synchronous case is used. As such, the default constructor uses a
synchronous schedule.

$(FIELDS)

Systems that include the model semantics wrap around this struct with an
[`ArbitrarySteppable`](https://juliadynamics.github.io/DynamicalSystems.jl/stable/tutorial/#DynamicalSystemsBase.ArbitrarySteppable)
from [`DynamicalSystems`](https://juliadynamics.github.io/DynamicalSystems.jl/stable/). See
[`create_qn_system`](@ref) for an example.
"""
struct QualitativeNetwork{
    N,
    Schedule,
    M<:MetaGraph{Int,<:SimpleDiGraph,<:EntityLabel,<:Entity},
    
} <: GraphDynamicalSystem{N,Schedule}
    "Graph containing the topology and target functions of the network"
    graph::M
    "State of the network"
    state::MVector{N,Int}

    function QualitativeNetwork(graph, state; schedule = Synchronous)
        N = nv(graph)
        if N != length(state)
            error("""The number of entities in the model ($N) must match the \
                  length of the provided state vector ($(length(state))).""")
        end

        return new{N,schedule(),typeof(graph)}(graph, state)
    end
end

function QualitativeNetwork(
    entities::AbstractVector{<:Entity};
    state = nothing,
    schedule = Synchronous,
)
    graph = update_functions_to_interaction_graph(entities; schedule)

    if isnothing(state)
        state = rand.(domain.(entities))
    end

    return QualitativeNetwork(graph, state; schedule)
end

QualitativeNetwork(entities::AbstractVector{<:AbstractString}, args...; kwargs...) =
    QualitativeNetwork(EntityName.(Symbol.(entities)), args...; kwargs...)

"""
    $(TYPEDSIGNATURES)

Shorthand for [`QualitativeNetwork`](@ref).
"""
const QN = QualitativeNetwork

"""
    $(TYPEDSIGNATURES)

Get all entities of the QN.
"""
function get_entities(qn::QN)
    return [qn.graph[e] for e in  get_entity_names(qn)]
end

"""
    $(TYPEDSIGNATURES)

wrapper for symbols
"""
function get_domain(qn::QN, entity_label::Symbol)
    return get_domain(qn, EntityName(entity_label))
end

"""
    $(TYPEDSIGNATURES)

Get the domain of the entity `entity_label` in `qn`.
"""
function get_domain(qn::QN, entity_label::EntityName)
    graph = get_graph(qn)
    entity = graph[entity_label]

    return domain(entity)
end

function get_domain(qn::QN, entity::Entity)
    return get_domain(qn, entity.label)
end

"""
    $(TYPEDSIGNATURES)

Get all of the domains of the entities in `qn`.
"""
function get_domain(qn::QN)
    return get_domain.((qn,), get_entity_names(qn))
end

function _get_entity_index(qn::QN, entity::Union{Entity, EntityLabel})
    i = if entity isa EntityLabel
        findfirst(==(entity), get_entity_names(qn))
    elseif entity isa Entity
        findfirst(==(entity.label), get_entity_names(qn))
    end
    if isnothing(i)
        error("""Tried to get the state of $entity but could not retrieve it. \
              The entities in the model are $(get_entities(qn))""")
    end
    return i
end

"""
    $(TYPEDSIGNATURES)
"""
function target_functions(qn::QN)
    return Dict([
        c => target_function(entity) for (c, (_, entity)) in get_graph(qn).vertex_properties
    ])
end

"""
    $(TYPEDSIGNATURES)
"""
function get_state(qn::QN, entity::Entity)
    i = _get_entity_index(qn, entity)
    return qn.state[i]
end

function get_state(qn::QN, entity_name::EntityName)
    i = _get_entity_index(qn, entity_name)
    return qn.state[i]
end 


function _set_state!(qn::QN, entity, value::Integer)
    i = _get_entity_index(qn::QN, entity)
    qn.state[i] = value
end

"""
    $(TYPEDSIGNATURES)
"""
function set_state!(qn::QN, entity::Entity, value::Integer)
    max_for_entity = maximum(domain(entity))
    if value > max_for_entity
        error(
            "Value ($value) cannot be larger than the maximum level for $entity ($(max_for_entity))",
        )
    end

    _set_state!(qn, entity, value)
end

function set_state!(qn::QN, entity_name::EntityName, value::Integer)
    set_state!(qn, qn.graph[entity_name], value)
end

function set_state!(qn::QN, values)
    set_state!.((qn,), get_entities(qn), values)
end

"""
    $(TYPEDSIGNATURES)

Interpret target functions from a [`QualitativeNetwork`](@ref).
"""
function interpret(e::Union{Expr,EntityName, Symbol,Int}, qn::QN, target)
    @info "expresion $e"
    function scale_with_domain(source::EntityName, target::Union{EntityName, Entity}, val::Integer)
        (source_min, source_max) = extrema(get_domain(qn, source))
        (target_min, target_max) = extrema(get_domain(qn, target))
        if (source_max == source_min)
            return source_mi
        else
            return(Integer(round(
                (val - source_min)*(
                (target_max-target_min)/(source_max-source_min)
                )+ target_min
                )))
        end
    end
    @match e begin
        ::Symbol => scale_with_domain(EntityName(e), target, get_state(qn, EntityName(e)))
        ::EntityName => scale_with_domain(e, target, get_state(qn, e))
        ::Int => e
        :($v1 + $v2) => interpret(v1, qn, target) + interpret(v2, qn, target)
        :($v1 - $v2) => interpret(v1, qn, target) - interpret(v2, qn, target)
        :($v1 / $v2) => interpret(v1, qn, target) / interpret(v2, qn, target)
        :($v1 * $v2) => begin
            r1, r2 = interpret(v1, qn, target), interpret(v2, qn, target)
            r1 < 0 && r2 < 0 ? 0 : r1*r2
        end
        :(min($v1, $v2)) => min(interpret(v1, qn, target), interpret(v2, qn, target))
        :(max($v1, $v2)) => max(interpret(v1, qn, target), interpret(v2, qn, target))
        :(ceil($v)) => ceil(interpret(v, qn, target))
        :(floor($v)) => floor(interpret(v, qn, target))
        _ => error("Unhandled Expr in `interpret`: $e")
    end
end


"""
    $(TYPEDSIGNATURES)

Returns the limited value of `next_value` which is at most 1 different than `prev_value`.

It is also never negative, or larger than `N`.
"""
function limit_change(
    prev_value::Integer,
    next_value::Number,
    min_level::Integer,
    max_level::Integer,
)
    if next_value > prev_value
        limited_value = min(prev_value + 1, max_level)
    elseif next_value < prev_value
        limited_value = max(prev_value - 1, min_level)
    else
        limited_value = next_value
    end

    return limited_value
end

"""
    $(TYPEDSIGNATURES)

Returns the limited value of `next_value` which is at most 1 different than `prev_value`.

It is also never negative, or larger than `N`.
"""
function limit_change(entity::Entity, prev_value::Integer, next_value::Number)::Integer
    min_level, max_level = range_from(entity), range_to(entity)
    return limit_change(prev_value, next_value, min_level, max_level)
    
end

function _compute_next_state!(qn::QN, entity::EntityName)
    (min_level, max_level) = extrema(get_domain(qn, entity))
    t = target_functions(qn)[entity]
    old_state = get_state(qn, entity)
    new_state = interpret(t, qn, entity)
    new_state = isnan(new_state) ? min_level : new_state
    new_state = isinf(new_state) ? max_level : new_state
    limited_state = limit_change(old_state, floor(Int, new_state), min_level, max_level)
end

function _compute_next_state!(qn::QN, entity::Entity)
    _compute_next_state!(qn, entity.label)
end

"""
    $(TYPEDSIGNATURES)
"""
function async_qn_step!(qn::QN)
    entity_labels = get_entities(qn)
    entity = rand(entity_labels)
    next_state = _compute_next_state!(qn, entity)
    set_state!(qn, entity, next_state)
end

"""
    $(TYPEDSIGNATURES)
"""
function sync_qn_step!(qn::QN)
    next_states = _compute_next_state!.((qn,), get_entities(qn))
    set_state!.((qn,), get_entities(qn), next_states)
end

"""
    $(TYPEDSIGNATURES)
"""
function qn_step!(qn::QN)
    if get_schedule(qn) == Asynchronous()
        async_qn_step!(qn)
    else
        sync_qn_step!(qn)
    end
end

"""
    $(TYPEDSIGNATURES)
"""
function get_step_function(qn::QN)
    return get_schedule(qn) == Asynchronous() ? async_qn_step! : sync_qn_step!
end

extract_state(model::QN) = model.state
extract_parameters(model::QN) = model.graph
current_parameters(model::QN) = model.graph
reset_model!(model::QN, u, _) = model.state .= u

function SciMLBase.reinit!(
    ds::ArbitrarySteppable{<:AbstractVector{<:Real},<:QualitativeNetwork},
    u::AbstractVector{<:Real} = initial_state(ds);
    p = current_parameters(ds),
    t0 = 0, # t0 is not used but required for downstream.
)
    ds.reinit(ds.model, u, p)
    ds.t[] = 0
    return ds
end

"""
    $(TYPEDSIGNATURES)

Construct an asynchronous [`QualitativeNetwork`](@ref) system using the
[`async_qn_step!`](@ref) as a step function.
"""
function create_qn_system(qn::QN)
    step_fn = get_schedule(qn) == Asynchronous() ? async_qn_step! : sync_qn_step!

    return ArbitrarySteppable(
        qn,
        step_fn,
        extract_state,
        extract_parameters,
        reset_model!,
        isdeterministic = get_schedule(qn) == Synchronous(),
    )
end


"""
    $(TYPEDSIGNATURES)

Given a function, classify entities in the function into activators and inhibitors.
"""
function classify_activators_inhibitors(
    ex,
    sign::Int = 1,
    activators::AbstractVector = EntityName[],
    inhibitors::AbstractVector = EntityName[],
)
    (activators, inhibitors) = @match ex begin
        ::EntityName => if sign == 1
            (push!(activators, ex), inhibitors)
        else
            (activators, push!(inhibitors, ex))
        end
        ::Int => (activators, inhibitors)
        Expr(:call, :(-), child) =>
            classify_activators_inhibitors(child, -sign, activators, inhibitors)
        Expr(:call, :(-), left_child, right_child) => begin
            (activators, inhibitors) = classify_activators_inhibitors(
                left_child,
                sign,
                activators,
                inhibitors,
            )
            (activators, inhibitors) = classify_activators_inhibitors(
                right_child,
                -sign,
                activators,
                inhibitors,
            )
            (activators, inhibitors)
        end
        Expr(:call, f, children...) => begin
            for child in children
                (activators, inhibitors) = classify_activators_inhibitors(
                    child,
                    sign,
                    activators,
                    inhibitors,
                )
            end
            (activators, inhibitors)
        end
        Expr(expr_type, _...) => error("Can't classify expression of type $expr_type")
    end

    return activators, inhibitors
end
