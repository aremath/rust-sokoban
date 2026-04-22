// Perform Monte-Carlo Tree Search on SokoStates
// Includes a Searchable trait; MCTS is implemented over everything Searchable
// Aslo includes BFS
#![feature(file_buffered)]
use std::fs::File;
use std::io::Write;
use std::io::BufWriter;
use std::collections::HashMap;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::cmp::Ordering;
use std::hash::Hash;
use ordered_float::{OrderedFloat, Pow};
use strum::IntoEnumIterator;
use std::collections::VecDeque;
use std::time::Instant;
use rand::prelude::*;
use rand::rngs::SmallRng;
use std::f64::consts::E as eulers_constant;

use crate::sokoengine::{SokoManager, SokoState, HasVecs, MapTile, Entity, Direction, SokoInterface, Stringable, Coder, IsPlayer};
use crate::sokoset::{SokoSet, SetManager};

#[derive(Debug, Copy, Clone)]
pub enum SelectPolicy {
    EpsilonGreedy,
    Softmax,
}

//TODO: move more stuff into SearchSettings
// e.g. the heuristic
#[derive(Debug, Clone)]
pub struct SearchSettings {
    // Multiplier on ln_Ni inside the exploration term
    // Higher means more priority to explore
    exploration_bonus: OrderedFloat<f64>, //TODO: how to determine this?
    // Scale factor on the exploitation (long-term heuristic value) term
    // Higher means more priority to follow the heuristic
    exploitation_scale: OrderedFloat<f64>,
    // 0-1 mixture between max and average long-term heuristic value
    // 0.0 means only average, 1.0 means only maximum
    maximization_bias: OrderedFloat<f64>,
    // % of the time that the "optimal" action is taken greedily when using eps-greedy
    // 0.6 means that the highest-long-term-value action will be taken 60% of the time
    epsilon: OrderedFloat<f64>,
    // Scale factor for the "inherent" value of a state (the heuristic value for the state itself)
    inherent_value_scale: OrderedFloat<f64>,
    select_policy : SelectPolicy,
    rollout_length: Option<usize>,
    n_rollouts_per_leaf: usize,
    gamma: OrderedFloat<f64>,
}

impl SearchSettings {

    pub fn new(exploration_bonus: OrderedFloat<f64>,
        exploitation_scale: OrderedFloat<f64>,
        maximization_bias: OrderedFloat<f64>,
        epsilon: OrderedFloat<f64>,
        inherent_value_scale: OrderedFloat<f64>,
        selection_policy: SelectPolicy,
        rollout_length: Option<usize>,
        n_rollouts_per_leaf: usize,
        gamma: OrderedFloat<f64>) -> Self {
            return SearchSettings { exploration_bonus : exploration_bonus,
                exploitation_scale : exploitation_scale,
                maximization_bias : maximization_bias,
                epsilon : epsilon,
                inherent_value_scale : inherent_value_scale,
                select_policy : selection_policy,
                rollout_length : rollout_length,
                n_rollouts_per_leaf : n_rollouts_per_leaf,
                gamma : gamma,
            };
        }

}

pub trait Searchable {
    type V: HasVecs;
    // Action type
    type A;
    fn neighbors(&self, mgr: &Self::V) -> Vec<(Self::A, Self)> where Self: Sized;
    // Used to terminate rollouts
    fn terminal(&self) -> bool;
    fn is_win(&self) -> bool;
}

impl Searchable for SokoState<MapTile, Entity> {
    type V = SokoManager<MapTile, Entity>;
    type A = Direction;

    fn neighbors(&self, mgr: &SokoManager<MapTile, Entity>) -> Vec<(Self::A, SokoState<MapTile, Entity>)> {
        let mut v = Vec::new();
        /*
        // Winning state has no neighbors
        if self.is_win() {
            return v;
        }
        */
        for d in Direction::iter() {
            match self.update(d, mgr) {
                Some(s) => { v.push((d,s)) },
                None => {}
            }
        }
        return v;
    }

    fn terminal(&self) -> bool {
        return SokoInterface::is_win(self);
    }

    fn is_win(&self) -> bool {
        return SokoInterface::is_win(self);
    }
}

impl Searchable for SokoSet {
    type V = SetManager;
    type A = Direction;

    fn neighbors(&self, mgr: &SetManager) -> Vec<(Self::A, SokoSet)> {
        let v = Vec::new();
        return v; //TODO
    }

    fn terminal(&self) -> bool {
        match self.resolve_singleton() {
            Some(_) => { true },
            None => { false }
        }
    }

    // TODO: you can't "win" an in-progress design (?)
    fn is_win(&self) -> bool {
        match self.resolve_singleton() {
            Some(s) => { SokoInterface::is_win(&s) },
            None => { false }
        }
    }
}

// "time" tags enforce DAGness of the state graph
pub type Tagged<T> = (T, usize);
pub type TaggedSokoState<M, E> = Tagged<SokoState<M, E>>;

impl<T> Searchable for Tagged<T> where
    T: Searchable
{
    type V = <T as Searchable>::V;
    type A = <T as Searchable>::A;

    fn neighbors(&self, mgr: &Self::V) -> Vec<(Self::A, Tagged<T>)> {
        let mut v = Vec::new();
        for (a, n) in self.0.neighbors(mgr) {
            v.push((a, (n, self.1 + 1)));
        }
        return v
    }

    fn terminal(&self) -> bool {
        return Searchable::terminal(&self.0);
    }

    fn is_win(&self) -> bool {
        return Searchable::is_win(&self.0);
    }
}

impl<M: Eq + Hash + Copy + Default, E: Eq + Hash + Copy + Default + IsPlayer, C: Coder<M, E>> Stringable<M, E, C> for TaggedSokoState<M, E> {
    fn from_str(s: &String, mgr: &C) -> Self {
        return (SokoState::<M, E>::from_str(s, mgr), 0);
    }

    fn to_str(&self, mgr: &C) -> String {
        return self.0.to_str(mgr);
    }
}

#[derive(Clone, Eq, PartialEq)]
struct SearchState<T: Clone + Eq + PartialEq> {
    p: OrderedFloat<f64>, // Priority
    // Break ties using the time of discovery of the state
    state_id: usize,
    t: T,
}

struct StateData {
    // Combination of child values
    value_estimate: OrderedFloat<f64>, // "n_wins" in traditional MCTS
    max_value: OrderedFloat<f64>, // The maximum value we've seen from a descendant of this node
    inherent_value: OrderedFloat<f64>, // The heuristic value for the state itself
    n_samples: usize,
}

impl StateData {

    pub fn new(h: OrderedFloat<f64>) -> Self {
        return StateData { value_estimate: OrderedFloat(0.0),
            max_value: OrderedFloat(0.0),
            inherent_value: h,
            n_samples: 0 };
    }

}

impl<T: Clone + Eq + PartialEq> Ord for SearchState<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.p.cmp(&other.p).then_with(|| self.state_id.cmp(&other.state_id))
    }
}

impl<T: Clone + Eq + PartialEq> PartialOrd for SearchState<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct SearchTree<T: Clone + Eq + PartialEq + Hash + Searchable> {
    pub initial_state: T,
    pub t: usize,
    settings: SearchSettings,
    //rng: SmallRng,
    visited: Box<HashSet<T>>,
    visits_this_episode: Box<HashMap<T, usize>>,
    single_parents: Box<HashMap<T, T>>,
    parents: Box<HashMap<T, Vec<T>>>,
    children: Box<HashMap<T, Vec<T>>>,
    state_data: Box<HashMap<T, StateData>>,
    state_queue: Box<BinaryHeap<SearchState<T>>>,
}

impl<T: Clone + Eq + PartialEq + Hash + Searchable> SearchTree<T> where
    T: Stringable<MapTile, Entity, <T as Searchable>::V>
{

    pub fn new(initial_state: T, settings: SearchSettings) -> Self {
        let h = HashMap::new();
        let parents = HashMap::new();
        let mut v = HashSet::new();
        let mut p = BinaryHeap::new();
        let d = HashMap::new();
        let c = HashMap::new();
        //let rng = SmallRng::seed_from_u64(SEED_VALUE);
        v.insert(initial_state.clone());
        let s0 = SearchState { p: OrderedFloat(0.0), state_id: 0, t: initial_state.clone() };
        p.push(s0);
        return SearchTree { initial_state: initial_state.clone(),
            t: 1,
            settings: settings,
            //rng : rng,
            visited : Box::new(v),
            visits_this_episode : Box::new(HashMap::new()),
            parents: Box::new(parents),
            single_parents: Box::new(h),
            children: Box::new(c),
            state_data: Box::new(d),
            state_queue : Box::new(p) };
    }


    fn exploration_score(&self, data: &StateData, parent_n: usize) -> OrderedFloat<f64> {
        // parent_n = 0 means that the parent is a leaf node.
        //TODO: in practice, leaf nodes should never be evaluated this way, so may want to assert
        if parent_n == 0 {
            return OrderedFloat(0.0);
        }
        //TODO: does the +0.1 break things?
        // +0.1 so that the ratio is nonzero when n_samples is 1
        // Weighted sampling requires nonzero weights!
        let ln_Ni = OrderedFloat(f64::ln(parent_n as f64)) + 0.1;
        //TODO: does the +1 break things?
        // +1 so the ratio is not unbounded for nodes with 0 samples
        let ni = OrderedFloat(data.n_samples as f64 + 0.1);
        let ratio = ((self.settings.exploration_bonus * ln_Ni) / ni).pow(0.5);
        return ratio;
    }

    fn exploitation_score(&self, data: &StateData) -> OrderedFloat<f64> {
        let average_h = data.value_estimate / (OrderedFloat(data.n_samples as f64) + 0.1);
        //println!("{}, {}", data.value_estimate, average_h);
        let max_h = data.max_value;
        let mixture = average_h * (OrderedFloat(1.0) - self.settings.maximization_bias) + max_h * self.settings.maximization_bias;
        return self.settings.exploitation_scale * mixture;
    }

    fn inherent_score(&self, data: &StateData) -> OrderedFloat<f64> {
        let i = data.inherent_value;
        return self.settings.inherent_value_scale * i;
    }

    //TODO: (how much) maximization bias is good?
    // e.g. have a max score then the value is some mixture of the average and the max
    //TODO: what about the heuristic score for the state itself?
    //  The issue with this is that it should decay for older (higher in the tree) states because the score is
    // likely to be less accurate, but how should it decay?
    // Compute the value estimate for a state
    //TODO: Caching this result in the data object could be more efficient
    // (but having to re-update every time a child pushes data might make it less efficient?)
    pub fn evaluate_state(&self, state: &T, parent_n: usize) -> OrderedFloat<f64> {
        let data = self.state_data.get(state).expect("State not found!");
        return self.exploration_score(data, parent_n) + self.exploitation_score(data) + self.inherent_score(data);
    }

    pub fn n_states(&self) -> usize {
        return self.visited.len();
    }

    // Unlike tree_select, rollout has to mutate self because it will add to the state tree
    // So we cannot have current_state as an immutable reference to self
    // Instead, we'll clone everything like usual...
    pub fn rollout(&mut self, start_state: &T, rollout_length: Option<usize>, rng: &mut SmallRng,
            heuristic: impl Fn(&T) -> OrderedFloat<f64>,
            mgr: &<T as Searchable>::V) -> (T, Vec<T>, usize) {
        let mut n_iters = 0;
        let mut trajectory = Vec::new();
        // Don't push the start state -- tree_select has leaf as the last state
        // So this function will only include proper descendants of the start state in the trajectory
        //trajectory.push(start_state.clone());
        let mut current_state = start_state.clone();
        loop {
            // Terminate the loop if we have come to the end of the state space
            if current_state.terminal() {
                break;
            }
            // Terminate the loop if the rollout is too long
            match rollout_length {
                Some(l) => { if n_iters >= l {
                    break;
                    }},
                None => { },
            }
            // Add this state's children to the tree
            self.add_children(&current_state, &heuristic, mgr);
            // Select a state at random from the current state's neighbors
            let child = self.select_child_random(rng, &current_state);
            match child {
                Some(c) => {
                    trajectory.push(c.clone());
                    current_state = c.clone();
                    n_iters += 1;
                }
                // TODO: may need different "reasons" to terminate
                // If current_state is a dead end but not winning, then we do NOT want to come back here,
                // regardless of what the heuristic says
                // Terminate the loop if current_state is a dead end
                None => {
                    break;
                }
            }
        }
        return (current_state, trajectory, n_iters); //TODO is there anything special needed if no iterations were run?
    }

    // Perform a rollout of a given length from a given start state
    //TODO: early stopping for dead states during the rollout?
    pub fn rollout_forget(&self, start_state: &T, rollout_length: Option<usize>, rng: &mut SmallRng, mgr: &<T as Searchable>::V) -> (T, Vec<T>, usize) {
        let mut n_iters = 0;
        let mut current_state = start_state.clone();
        loop {
            // Terminate the loop if we have come to the end of the state space
            if current_state.terminal() {
                break;
            }
            // Terminate the loop if the rollout is too long
            match rollout_length {
                Some(l) => { if n_iters >= l {
                    break;
                    }},
                None => { },
            }
            // Select a state at random from the current state's neighbors
            let mut neighbors = current_state.neighbors(mgr);
            if neighbors.len() == 0 {
                break;
            }
            // Have to choose the index because if I choose directly from the vector I will get a reference
            let indices = 0..neighbors.len();
            let child_index = indices.choose(rng).expect("No children states!");
            // Take ownership from the now-useless neighbors
            (_, current_state) = neighbors.swap_remove(child_index);
            n_iters += 1;
        }
        // Return an empty Vec to preserve cross-compatibility with the non-forgetting rollout
        return (current_state, Vec::new(), n_iters); //TODO is there anything special needed if no iterations were run?
    }

    // Depth-first upwards traversal to propagate value to all ancestors
    // Static method because state_data is borrowed mutably but parents is borrowed immutably
    //TODO: it is highly problematic to propagate to all ancestors.
    // Imagine:
    // A -> B -----> D
    //  --------> C /
    // Say D is very good.
    // If we traverse A -> B -> D and propagate upwards from D, then C gets credit for the A -> B -> D traversal
    // If D is very difficult to reach from C (or unlikely for some other reason), then the probability of C will
    // be boosted because of D, despite the fact that exploring C more will not have a significant effect on the
    // probability of going to D.
    fn propagate_upwards(state_data: &mut HashMap<T, StateData>,
            parents: &HashMap<T, Vec<T>>,
            state: &T, visited: &mut HashSet<T>, h: OrderedFloat<f64>) -> () {
        visited.insert(state.clone());
        let data = state_data.get_mut(state).expect("State not found!");
        data.value_estimate += h;
        if h > data.max_value {
            data.max_value = h;
        }
        data.n_samples += 1;
        match parents.get(state) {
            Some(parent_v) => {
                for parent in parent_v {
                    if !visited.contains(parent) {
                        SearchTree::propagate_upwards(state_data, parents, parent, visited, h);
                    }
                }
            },
            None => {}
        }
    }

    pub fn update_data(&mut self, state: &T, h: OrderedFloat<f64>) -> () {
        let data = self.state_data.get_mut(state).expect("State not found!");
        data.value_estimate += h;
        if h > data.max_value {
            data.max_value = h;
        }
        data.n_samples += 1;
        if !self.visits_this_episode.contains_key(state) {
            self.visits_this_episode.insert(state.clone(), 1);
        } else {
            let old = self.visits_this_episode.get(state).expect("State not found!");
            self.visits_this_episode.insert(state.clone(), old + 1);
        }
    }

    // Also backpropagates information gained during the rollout to the start_state's ancestors
    //TODO: currently no mechanism for early stopping if the rolled out state is actually a victory
    //  because rollout states are not saved, we don't know how to recover the victory path
    //TODO: tainted because of propagate_upwards
    pub fn rollout_backprop(&mut self, start_state: &T, rollout_length: Option<usize>,
        mgr: &<T as Searchable>::V, rng: &mut SmallRng, heuristic: impl Fn(&T) -> OrderedFloat<f64> ) -> () {
        // Do the rollout
        //println!("\tDo the rollout!");
        let (rolled_out, _, _) = self.rollout_forget(start_state, rollout_length, rng, mgr);
        //println!("Rolled out to\n{}", rolled_out.to_str(mgr));
        let h = heuristic(&rolled_out);
        //println!("\tPropagating {}", h);
        //println!("\tPropagate the heuristic!");
        // Propagate the heuristic value back up the tree
        let mut visited = HashSet::new();
        SearchTree::propagate_upwards(&mut self.state_data, &self.parents, start_state, &mut visited, h);
    }

    //TODO: convert the other select_child functions to work with select_child_generic
    pub fn select_child_generic<'a>(&'a self, current_state: &'a T, mut selector: impl FnMut(&T, &'a Vec<T>) -> Option<&'a T>) -> Option<&'a T> {
        let children_q = self.children.get(current_state);
        match children_q {
            Some(children) => {
                // None if we're at a previously-discovered dead-end
                if children.len() == 0 {
                    return None;
                }
                return selector(current_state, &children);
            }
            // None if we're at an undiscovered node
            None => {
                return None;
            }
        }
    }

    // Other tree selection algs are possible of course
    // have to separate the mutable borrow of rng from the immutable borrow of other variables
    pub fn softmax_child_selector<'a>(&self, current_state: &T, children: &'a Vec<T>, rng: &mut SmallRng) -> Option<&'a T> {
        let parent_n = self.state_data.get(current_state).expect("State not found!").n_samples;
        // Compute the softmax weights
        let scores = (&children).into_iter().map(|c| self.evaluate_state(c, parent_n));
        // First, convert to proportions
        // We convert to proportions first because the value estimates are usually quite small
        // Softmax does not really work well with small values because the derivative of e^x at 0 is 1
        // This means that softmax(x,y,z) ~ (x, y, z) for x,y,z ~ 0, so the softmax is basically doing nothing
        let p_sum: OrderedFloat<f64> = scores.clone().sum();
        let p_prob = scores.map(|s| (f64::from(s) * 10.0) / f64::from(p_sum));
        // Then, softmax the proportions
        let e_scores = p_prob.map(|s| eulers_constant.pow(f64::from(s)));
        let e_sum: f64 = e_scores.clone().sum();
        let e_prob = e_scores.map(|s| s / e_sum);
        // Zip them for convenient closure
        let cc: Vec<(&T, f64)> = children.into_iter().zip(e_prob).collect();
        /*
        for c in &cc {
            let data = self.state_data.get(c.0).expect("State not found!");
            let expl_a = self.exploration_score(data, parent_n);
            let expl_b = self.exploitation_score(data);
            println!("{} | {} = {} + {}", c.1, expl_a + expl_b, expl_a, expl_b);
        }
        */
        return Some(cc.choose_weighted(rng, |item| item.1).unwrap().0);
    }

    pub fn best_child_selector<'a>(&self, children: &'a Vec<T>) -> Option<&'a T> {
        return children.into_iter().max_by_key(|c| {
            let d = self.state_data.get(c).expect("State not found!");
            self.exploitation_score(d)
        });
    }

    // Select the child with the most samples in this episode
    //TODO: why do samples from previous episodes mess this up?
    pub fn select_child_most_sampled<'a>(&'a self, current_state: &'a T) -> Option<&'a T> {
        let children_q = self.children.get(current_state);
        match children_q {
            Some(children) => {
                // None if we're at a previously-discovered dead-end
                if children.len() == 0 {
                    return None;
                }
                let most_sampled_child = (&children).into_iter().max_by_key(|c| {
                    let d = self.state_data.get(c).expect("State not found!");
                    let n = match self.visits_this_episode.get(c) {
                        Some(k) => { k },
                        None => { &0 }
                    };
                    println!("{} -> {} | {}", d.n_samples, n, self.exploitation_score(d));
                    //d.n_samples
                    n});
                return most_sampled_child;
            }
            // None if we're at an undiscovered node
            None => {
                return None;
            }
        }
    }

    pub fn select_child_random<'a>(&'a self, rng: &mut SmallRng, current_state: &'a T) -> Option<&'a T> {
        let children_q = self.children.get(current_state);
        match children_q {
            Some(children) => {
                // None if we're at a previously-discovered dead-end
                if children.len() == 0 {
                    return None;
                }
                // Otherwise, choose randomly
                return Some(children.choose(rng).unwrap());
            }
            // None if we're at an undiscovered node
            None => {
                return None;
            }
        }
    }

    //TODO: "per node" epsilon greedy, with epsilon set based on the empirical likelihood of finding value by following the greedy strategy
    pub fn select_child_epsilon_greedy<'a>(&'a self, rng: &mut SmallRng, current_state: &'a T) -> Option<&'a T> {
        let parent_n = self.state_data.get(current_state).expect("State not found!").n_samples;
        let children_q = self.children.get(current_state);
        match children_q {
            Some(children) => {
                // None if we're at a previously-discovered dead-end
                if children.len() == 0 {
                    return None;
                }
                // Greedy option
                if OrderedFloat(rng.random::<f64>()) < self.settings.epsilon {
                    //let scores = (&children).into_iter().map(|c| self.evaluate_state(c, parent_n));
                    // Just use the exploitation score
                    let scores = (&children).into_iter().map(|c| {
                        let d = self.state_data.get(c).expect("State not found!");
                        self.exploitation_score(d)});
                    let cc: Vec<(&T, OrderedFloat<f64>)> = children.into_iter().zip(scores).collect();
                    return Some(cc.iter().max_by_key(|c| c.1).unwrap().0);
                }
                // Otherwise, choose randomly
                return Some(children.choose(rng).unwrap());
            }
            // None if we're at an undiscovered node
            None => {
                return None;
            }
        }
    }

    // Need this lifetime annotation because this might return a reference from either
    // a) self interior through following the child connections OR
    // b) root from just putting root
    //TODO: can fix this by having the SearchTree just keep a copy of the root
    pub fn tree_select_leaf(&self, rng: &mut SmallRng) -> (&T, Vec<T>) {
        let mut trajectory = Vec::new();
        trajectory.push(self.initial_state.clone());
        let mut state = &self.initial_state;
        let mut layer = 0;
        // Traverse down the tree via tree_select until the state is a leaf
        loop {
            //println!("Layer: {}", layer);
            layer += 1;
            let child = match self.settings.select_policy {
                SelectPolicy::EpsilonGreedy => { self.select_child_epsilon_greedy(rng, &state) },
                SelectPolicy::Softmax => { self.select_child_generic(&state, |s, c| self.softmax_child_selector(s, c, rng)) }
            };
            match child {
                Some(c) => { 
                    trajectory.push(c.clone());
                    state = c; },
                None => { return (state, trajectory); }
            }
        }
    }

    /*
    //TODO
    // Descend the tree, until we reach a leaf, keeping track of where we have been to avoid cycles
    // Returns an Option because it is possible to go in a spiral pattern
    // and reach a node whose neighbors are all cycles, but the node is not a leaf
    //TODO: a better version of this would use DFS to find a valid leaf
    // But this would require re-engineering the child selection functions to produce a child ORDERING rather than
    // selecting a single child...
    pub fn tree_select_leaf_nondag(&self, rng: &mut SmallRng) -> Option((&T, Vec<T>)) {
        let mut trajectory = Vec::new();
        let mut visited = HashSet::new();
        trajectory.push(self.initial_state.clone());
        visited.insert(self.initial_state.clone());
        let mut state = &self.initial_state;
        let mut layer = 0;
        // Traverse down the tree via tree_select until the state is a leaf
        loop {
            //println!("Layer: {}", layer);
            layer += 1;
            let valid_children = 
            let children_q = self.children.get(current_state);
            match children_q {
                Some(children) => {
                    let valid_children = ??? TODO
                }
                None => {}
            }
            //TODO: have the select_child_generic take a function that determines the validity of possible children
            let child = match self.settings.select_policy {
                SelectPolicy::EpsilonGreedy => { self.select_child_generic(&state, |s, c| self.epsilon_greedy_child_selector(s, c, rng)) },
                SelectPolicy::Softmax => { self.select_child_generic(&state, |s, c| self.softmax_child_selector(s, c, rng) }
            };
            match child {
                Some(c) => { 
                    trajectory.push(c.clone());
                    state = c; },
                None => { return (state, trajectory); }
            }
        }
    }
    */

    fn is_leaf(&self, state: &T) -> bool {
        match self.children.get(state) {
            Some(_) => {false},
            None => {true},
        }
    }

    fn add_children(&mut self, state: &T, heuristic: impl Fn(&T) -> OrderedFloat<f64>,
        mgr: &<T as Searchable>::V) -> () {
        // This state's children were already added
        if self.children.contains_key(state) {
            return;
        }
        // Add all of state's children to the tree with empty node data, and track their parents
        self.children.insert(state.clone(), Vec::new());
        for (_, child) in state.neighbors(mgr) {
            //TODO: self.children must be a DAG or select_leaf may not terminate
            self.children.get_mut(&state).expect("Leaf not found!").push(child.clone());
            // Add the leaf to the neighbor's parents
            if !self.parents.contains_key(&child) {
                self.parents.insert(child.clone(), Vec::new());
            }
            self.parents.get_mut(&child).expect("Neighbor not found!").push(state.clone());
            // Every node has a unique parent, so only update parents for newly discovered nodes
            if !((self.visited).contains(&child)) {
                self.visited.insert(child.clone());
                self.single_parents.insert(child.clone(), state.clone());
                self.t += 1;
                //println!("{}", self.t);
                //println!("{}", neighbor == leaf);
                let h_c = heuristic(&child);
                self.state_data.insert(child.clone(), StateData::new(h_c));
            }
        }
    }

    //TODO: why not just select a node rather than selecting by traversing the tree?
    // If you keep a global queue of the states, then MCTS is guaranteed to eventually find a path if one exists
    //TODO: use until() instead of Searchable::is_win()?
    pub fn mcts(&mut self, heuristic: impl Fn(&T) -> OrderedFloat<f64>,
        max_states: Option<usize>, max_iters: Option<usize>, min_samples: Option<usize>,
        mgr: &<T as Searchable>::V,
        rng: &mut SmallRng) -> Option<T> {
        let mut n_iters = 0;
        // Give the starting state a default value
        // This will always happen for a blank tree, but if you start MCTS with a partial tree, then
        // we don't want to overwrite the value
        if !self.state_data.contains_key(&self.initial_state) {
            let h0 = heuristic(&self.initial_state);
            self.state_data.insert(self.initial_state.clone(), StateData::new(h0));
        }
        //println!("{}", self.parents.get(&self.initial_state).is_none());
        loop {
            //println!("Iteration {} Start!", n_iters);
            n_iters += 1;
            if let Some(m) = max_states && self.t > m {
                return None;
            }
            if let Some(m) = max_iters && n_iters > m {
                return None;
            }
            // If we have enough samples, we can stop
            if let Some(m) = min_samples {
                if self.state_data.get(&self.initial_state).expect("State not found!").n_samples > m {
                    return None;
                }
            }
            //println!("Step 1: select leaf");
            // Step 1, Tree selection policy to traverse from root to leaf
            let (leaf, trajectory) = self.tree_select_leaf(rng);
            // Drop the borrow via cloning (leaf is a reference into self.children, and we need to change that)
            let leaf = leaf.clone();
            //println!("Selected:\n{}", leaf.to_str(mgr));
            // Step 1.5, Check if the leaf node is a victory (or the new leaf node)
            if leaf.is_win() {
                // The unroll function can reproduce the path
                return Some(leaf.clone());
            }
            //println!("Step 1.8: Add children");
            // Add all of its children to the tree with empty node data, and track their parents
            self.add_children(&leaf, &heuristic, mgr);
            //println!("Step 3: Roll out");
            //println!("Root still no parents: {}", self.parents.get(&self.initial_state).is_none());
            // Step 2, Roll out from the leaf
            //TODO: should this roll out from one of the children?
            //self.rollout_backprop(&leaf, rollout_length, mgr, rng, &heuristic);
            //let rolled_out = self.rollout_forget(&leaf, rollout_length, rng, mgr);
            let mut n_rollouts = 0;
                while n_rollouts < self.settings.n_rollouts_per_leaf {
                n_rollouts += 1;
                let (rolled_out, trajectory2, depth) = self.rollout(&leaf, self.settings.rollout_length, rng, &heuristic, mgr);
                if rolled_out.is_win() {
                    return Some(rolled_out.clone());
                }
                //println!("Rolled out to\n{}", rolled_out.to_str(mgr));
                let h = heuristic(&rolled_out);
                let mut current_gamma = OrderedFloat(1.0);
                //println!("\tPropagating {}", h);
                // Step 3, propagate the rollout score up the tree along the trajectory
                // First, update the value for nodes discovered during the rollout
                // Reverse because the node closest to the leaf should get the highest payout!
                for ancestor2 in trajectory2.iter().rev() {
                    self.update_data(&ancestor2, h * current_gamma);
                    current_gamma = current_gamma * self.settings.gamma;
                }
                // Then update ancestors on the path we took to get here
                for ancestor in trajectory.iter().rev() {
                    //println!("Ancestor");
                    self.update_data(&ancestor, h * current_gamma);
                    current_gamma = current_gamma * self.settings.gamma;
                }
            }
        }
    }

    // Run MCTS for n rollouts, then commit to the "best" state and keep going
    //TODO: what is the right "best" strategy?
    //TODO: cut out part of the tree after committing?
    // Only keep the states in the subtree we decided to go to
    // Also have to keep the direct ancestors of the subtree root for path reconstruction
    //TODO: is it okay to keep the value estimates after committing? They are "stale"...
    // i.e. value estimates we get from this cycle will be deeper into the tree and therefore more accurate
    // This suggests that using a gamma-value to scale the estimates is a good idea
    // Later value estimates will eventually win...?
    // But doesn't that also mean that you want to scale the n_samples as well?
    // i.e. scale the n_samples by the distance from the sampled state
    // We want some way for more recent (deeper) value estimates to eventually beat older (shallower) ones
    //TODO: time budget per episode?
    //TODO: episode length based on the total amount of sampling done for the children rather than fixed
    // on episode 2, you don't have to do as many rollouts because many of the states have already been sampled a lot
    pub fn mcts_episodic(&mut self, heuristic: impl Fn(&T) -> OrderedFloat<f64>,
        max_states: Option<usize>, max_iters_per: Option<usize>, min_samples: Option<usize>,
        mgr: &<T as Searchable>::V,
        rng: &mut SmallRng) -> Option<T> {
        let mut n_episodes = 0;
        loop {
            println!("Episode {} Start!", n_episodes);
            println!("Total {} states", self.t);
            n_episodes += 1;
            println!("State: \n{}", self.initial_state.to_str(mgr));
            let out = self.mcts(&heuristic, max_states, max_iters_per, min_samples, mgr, rng);
            match out {
                // MCTS succeeded! This is great; now we have found a path
                Some(state) => { return Some(state) },
                // MCTS failed. Now we need to commit to the best neighbor found by MCTS
                None => {
                    // We ran out of state budget without finding the goal :(
                    if let Some(m) = max_states && self.t > m {
                        return None;
                    }
                    //let best_child = self.select_child_most_sampled(&self.initial_state);
                    let best_child = self.select_child_generic(&self.initial_state, |_s, c| self.best_child_selector(c));
                    // Update the initial_state and run MCTS from here (using all of the same values)
                    self.initial_state = best_child.expect("State not found!").clone();
                }
            }
            self.clear_visits();
        }
    }

    pub fn clear_visits(&mut self) -> () {
        self.visits_this_episode = Box::new(HashMap::new());
    }

    // Returns a reference to the best state found so far
    pub fn best_so_far(&self) -> Option<&T> {
        let mut best_state = None;
        let mut max_h = OrderedFloat(0.0);
        for state in self.visited.iter() {
            let d = self.state_data.get(state);
            match d {
                Some(data) => {
                    if data.n_samples > 0 {
                        //println!("{}", f64::from(data.value_estimate));
                        let exploitation = self.exploitation_score(data);
                        if exploitation > max_h {
                            max_h = exploitation;
                            best_state = Some(state);
                        }
                    }
                },
                None => {}
            }
        }
        match best_state {
            Some(_s) => { println!("{}",
                f64::from(max_h)) },
            None => {}
        }
        return best_state;
    }

    pub fn best_heuristic_so_far(&self, heuristic: impl Fn(&T) -> OrderedFloat<f64>) -> Option<&T> {
        let mut best_state = None;
        let mut max_h = OrderedFloat(0.0);
        for state in self.visited.iter() {
            let h = heuristic(state);
            if h > max_h {
                max_h = h;
                best_state = Some(state);
            }
        }
        match best_state {
            Some(_s) => { println!("{}",
                f64::from(max_h)) },
            None => {}
        }
        return best_state;
    }

    // BFS
    //TODO: Stop all the cloning!
    // Instead, have visited hold the master list of discovered states,
    // and make everything else use references
    // Implement Hash and Eq for &SokoState via dereferencing
    // Does not use the standard self.state_queue, which is a priority minqueue
    pub fn basic_search(&mut self, until: Option<fn(&T) -> bool>, max_states: Option<usize>, mgr: &<T as Searchable>::V) -> Option<T> {
        let mut queue = VecDeque::new();
        queue.push_back(self.initial_state.clone());
        let mut time = Instant::now();
        let mut current_n: usize = 1;
        // Continue until the queue is empty
        while let Some(parent) = queue.pop_front() {
            //println!("{} states remain", queue.len());
            //println!("Current\n{}", parent.to_str(mgr));
            let neighbors = parent.neighbors(mgr);
            for (_action, neighbor) in neighbors {
                //println!("Child\n{}", neighbor.to_str(mgr));
                if !((*self.visited).contains(&neighbor)) {
                    self.visited.insert(neighbor.clone());
                    //println!("Added!");
                    self.single_parents.insert(neighbor.clone(), parent.clone());
                    //let n0 = SearchState { p: OrderedFloat(self.t as f64), state_id: self.t, t: neighbor.clone() };
                    self.t = self.t + 1;
                    if (self.n_states() % 10000) == 0 {
                        let ms = time.elapsed().as_millis();
                        let n_processed = self.n_states() - current_n;
                        let per = (n_processed as f64 / ms as f64) * 1000.0;
                        println!("{} states per second", per);
                        //println!("{}", ms);
                        println!("{} states total", self.n_states());
                        current_n = self.n_states();
                        time = Instant::now();
                    }
                    // Quit early if we've explored too many states
                    match max_states {
                        Some(m) => if self.t > m {
                                return None
                            },
                        None => {}

                    }
                    queue.push_back(neighbor.clone());
                    match until {
                        Some(f) => {if f(&neighbor) { 
                                return Some(neighbor);
                            }},
                        None => {}
                    }
                }
            }
        }
        return None;
    }

    // Find the path from start to end based on the search results
    //TODO: record and use the actions to recreate an action sequence
    pub fn unroll_search(&self, end_state: T, start_state: T) -> Option<Vec<T>> {
        let mut current_state = &end_state;
        let mut result = Vec::new();
        result.push(current_state.clone());
        while *current_state != start_state {
            match self.single_parents.get(current_state) {
                Some(s) => { current_state = s;
                        result.push(s.clone());
                    },
                None => { return None }
            }
        }
        result.reverse();
        return Some(result);
    }

    pub fn write_dag(&self, path_a: &str, path_b: &str) -> std::io::Result<()> {
        //TODO: BufWriter
        let mut f_a = BufWriter::new(File::create(path_a)?);
        let mut state_ids = HashMap::new();
        // Associate states with numbers
        //writeln!(&mut f_a, "Section: node_info");
        for (i, state) in self.visited.iter().enumerate() {
            state_ids.insert(state, i);
            let i_data = self.state_data.get(state).expect("State not found!");
            let i_n = i_data.n_samples;
            let i_e = self.exploitation_score(i_data);
            let i_d = i_data.inherent_value;
            let line = format!("{},{},{},{}", i, i_n, i_e, i_d);
            writeln!(&mut f_a, "{}", line);
        }
        f_a.flush()?;
        let mut f_b = BufWriter::new(File::create(path_b)?);
        writeln!(&mut f_b, "Source, Target\n");
        //writeln!(&mut f, "Section: node_edges");
        //TODO: self.parents is not correct?
        for parent in self.children.keys() {
            let parent_id = state_ids.get(parent).expect("State not found!");
            for child in self.children.get(parent).expect("State not found!") {
                let child_id = state_ids.get(child).expect("State not found!");
                // Edges from all parents to the child
                let line = format!("{},{}", parent_id, child_id);
                writeln!(&mut f_b, "{}", line);
            }
        }
        f_b.flush()?;
        return Ok(());
    }

}

//TODO: make this function polymorphic
pub fn mcts_forget (initial_state: SokoState<MapTile, Entity>,
        settings: SearchSettings,
        heuristic: impl Fn(&TaggedSokoState<MapTile, Entity>) -> OrderedFloat<f64>,
        max_states: Option<usize>, max_iters_per: Option<usize>, min_samples: Option<usize>,
        mgr: &<SokoState<MapTile, Entity> as Searchable>::V,
        rng: &mut SmallRng) -> Option<SokoState<MapTile, Entity>> {
        let mut n_episodes = 0;
        let mut s_init = initial_state.clone();
        loop {
            println!("Episode {} Start!", n_episodes);
            n_episodes += 1;
            println!("State: \n{}", s_init.to_str(mgr));
            let s_init_tagged = (s_init.clone(), 0);
            let mut tree = SearchTree::new(s_init_tagged, settings.clone());
            let out = tree.mcts(&heuristic, max_states, max_iters_per, min_samples, mgr, rng);
            match out {
                // MCTS succeeded! This is great; now we have found a path
                Some(state) => { return Some(state.0) },
                // MCTS failed. Now we need to commit to the best neighbor found by MCTS
                None => {
                    let best_child = tree.select_child_most_sampled(&tree.initial_state);
                    // Update the initial_state and run MCTS from here (using all of the same values)
                    s_init = best_child.expect("State not found!").0.clone();
                }
            }
        }
    }
