// Perform Monte-Carlo Tree Search on SokoStates
// Includes a Searchable trait; MCTS is implemented over everything Searchable
// Aslo includes BFS
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
use std::cell::Cell;
use std::f64::consts::E;

use crate::sokoengine::{SokoManager, SokoState, HasVecs, MapTile, Entity, Direction, SokoInterface, Stringable};

//const SEED_VALUE: u64 = 0;
const EXPLORATION_BONUS: OrderedFloat<f64> = OrderedFloat(5.0); //TODO: how to determine this?
//const OE: OrderedFloat<f64> = OrderedFloat(E);

pub trait Searchable {
    type V: HasVecs;
    fn neighbors(&self, mgr: &Self::V) -> Vec<Self> where Self: Sized;
    // Used to terminate rollouts
    fn terminal(&self) -> bool;
    fn is_win(&self) -> bool;
}

impl Searchable for SokoState<MapTile, Entity> {
    type V = SokoManager<MapTile, Entity>;

    fn neighbors(&self, mgr: &SokoManager<MapTile, Entity>) -> Vec<SokoState<MapTile, Entity>> {
        let mut v = Vec::new();
        /*
        // Winning state has no neighbors
        if self.is_win() {
            return v;
        }
        */
        for d in Direction::iter() {
            match self.update(d, mgr) {
                Some(s) => { v.push(s) },
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

#[derive(Clone, Eq, PartialEq)]
struct SearchState<T: Clone + Eq + PartialEq> {
    p: OrderedFloat<f64>,
    // Break ties using the time of discovery of the state
    state_id: usize,
    t: T,
}

struct StateData {
    // Combination of child values
    value_estimate: OrderedFloat<f64>, // "n_wins" in traditional MCTS
    n_samples: usize,
}

impl StateData {

    pub fn new() -> Self {
        return StateData { value_estimate: OrderedFloat(0.0), n_samples: 0 };
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
    initial_state: T,
    t: usize,
    //rng: SmallRng,
    visited: Box<HashSet<T>>,
    single_parents: Box<HashMap<T, T>>,
    parents: Box<HashMap<T, Vec<T>>>,
    children: Box<HashMap<T, Vec<T>>>,
    state_data: Box<HashMap<T, StateData>>,
    state_queue: Box<BinaryHeap<SearchState<T>>>,
}

impl<T: Clone + Eq + PartialEq + Hash + Searchable> SearchTree<T> where
    T: Stringable<MapTile, Entity, <T as Searchable>::V>
{

    pub fn new(initial_state: T) -> Self {
        let h = HashMap::new();
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
            //rng : rng,
            visited : Box::new(v),
            single_parents: Box::new(h),
            children: Box::new(c),
            state_data: Box::new(d),
            state_queue : Box::new(p) };
    }

    //TODO: (how much) maximization bias is good?
    // e.g. have a max score then the value is some mixture of the average and the max
    //TODO: what about the heuristic score for the state itself?
    //  The issue with this is that it should decay for older (higher in the tree) states because the score is
    // likely to be less accurate, but how should it decay?
    // Compute the value estimate for a state
    //TODO: Caching this result in the data object could be more efficient
    // (but having to re-update every time a child pushes data might make it less efficient?)
    pub fn evaluate_state(&self, state: &T) -> OrderedFloat<f64> {
        let data = self.state_data.get(state).expect("State not found!");
        let exploitation = data.value_estimate / (OrderedFloat(data.n_samples as f64) + 0.1);
        //TODO: theoretically this should be based on the sample ratio to the parent
        let exploration = EXPLORATION_BONUS / (OrderedFloat(data.n_samples as f64) + 1.0);
        return exploitation + exploration;
    }

    pub fn n_states(&self) -> usize {
        return self.visited.len();
    }

    // Perform a rollout of a given length from a given start state
    //TODO: early stopping for dead states during the rollout?
    pub fn rollout(&self, start_state: &T, rollout_length: Option<usize>, rng: &mut SmallRng, mgr: &<T as Searchable>::V) -> T {
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
            current_state = neighbors.swap_remove(child_index);
            n_iters += 1;
        }
        return current_state; //TODO is there anything special needed if no iterations were run?
    }

    // Depth-first upwards traversal to propagate value to all ancestors
    fn propagate_upwards(&mut self, state: &T, visited: &HashSet<T>, h: OrderedFloat<f64>) -> () {
        visited.add(state);
        let mut data = self.state_data.get_mut(state).expect("State not found!");
        data.value_estimate += h;
        data.n_samples += 1;
        match self.parents.get(state) {
            Some(parent_v) => {
                for parent in parent_v {
                    if !visited.contains(parent) {
                        self.propagate_upwards(parent);
                    }
                }
            },
            None => {}
        }
    }

    // Also backpropagates information gained during the rollout to the start_state's ancestors
    //TODO: currently no mechanism for early stopping if the rolled out state is actually a victory
    //  because rollout states are not saved, we don't know how to recover the victory path
    pub fn rollout_backprop(&mut self, start_state: &T, rollout_length: Option<usize>,
        mgr: &<T as Searchable>::V, rng: &mut SmallRng, heuristic: impl Fn(&T) -> OrderedFloat<f64> ) -> () {
        // Do the rollout
        println!("\tDo the rollout!");
        let rolled_out = self.rollout(start_state, rollout_length, rng, mgr);
        let h = heuristic(&rolled_out);
        println!("\tPropagate the heuristic!");
        // Propagate the heuristic value back up the tree
        let mut current_state = start_state;
        loop {
            let mut data = self.state_data.get_mut(current_state).expect("State not found!");
            data.value_estimate += h;
            data.n_samples += 1;
            match self.parents.get(current_state) {
                Some(p) => { current_state = p; },
                // If we get here, we're at the root
                None => { break; }
            }
        }
    }

    // Other tree selection algs are possible of course
    // have to separate the mutable borrow of rng from the immutable borrow of other variables
    pub fn tree_select_softmax<'a>(&'a self, rng: &mut SmallRng, current_state: &'a T) -> Option<&'a T> {
        //let children = self.children.get(current_state).expect("State not found!");
        let children_q = self.children.get(current_state);
        match children_q {
            Some(children) => {
                // None if we're at a previously-discovered dead-end
                if children.len() == 0 {
                    return None;
                }
                // Compute the softmax weights
                let scores = (&children).into_iter().map(|c| self.evaluate_state(c));
                let e_scores = scores.map(|s| E.pow(f64::from(s)));
                let e_sum: f64 = e_scores.clone().sum();
                let e_prob = e_scores.map(|s| s / e_sum);
                // Zip them for convenient closure
                let cc: Vec<(&T, f64)> = children.into_iter().zip(e_prob).collect();
                return Some(cc.choose_weighted(rng, |item| item.1).unwrap().0);
            }
            // None if we're at an undiscovered node
            None => {
                return None;
            }
        }
    }

    pub fn tree_select_epsilon_greedy(&mut self, current_state: &T) -> Option<&T> {
        return None // TODO
    }

    // Need this lifetime annotation because this might return a reference from either
    // a) self interior through following the child connections OR
    // b) root from just putting root
    //TODO: can fix this by having the SearchTree just keep a copy of the root
    pub fn tree_select_leaf(&self, rng: &mut SmallRng) -> &T {
        let mut state = &self.initial_state;
        let mut layer = 0;
        // Traverse down the tree via tree_select until the state is a leaf
        loop {
            println!("Layer: {}", layer);
            layer += 1;
            let child = self.tree_select_softmax(rng, &state);
            match child {
                Some(c) => { state = c; },
                None => { return state; }
            }
        }
    }

    fn is_leaf(&self, state: &T) -> bool {
        match self.children.get(state) {
            Some(_) => {false},
            None => {true},
        }
    }

    //TODO: why not just select a node rather than selecting by traversing the tree?
    // If you keep a global queue of the states, then MCTS is guaranteed to eventually find a path if one exists
    //TODO: use until() instead of Searchable::is_win()?
    pub fn mcts(&mut self, rollout_length: Option<usize>, heuristic: impl Fn(&T) -> OrderedFloat<f64>,
        max_states: Option<usize>, mgr: &<T as Searchable>::V,
        rng: &mut SmallRng) -> Option<T> {
        let mut n_iters = 0;
        // Give the starting state a default value
        self.state_data.insert(self.initial_state.clone(), StateData::new());
        println!("{}", self.parents.get(&self.initial_state).is_none());
        loop {
            println!("Iteration {} Start!", n_iters);
            n_iters += 1;
            if self.t > max_states? {
                return None;
            }
            println!("Step 1: select leaf");
            // Step 1, Tree selection policy to traverse from root to leaf
            let leaf = self.tree_select_leaf(rng).clone();
            // Step 1.5, Check if the leaf node is a victory (or the new leaf node)
            if leaf.is_win() {
                // The unroll function can reproduce the path
                return Some(leaf.clone());
            }
            println!("Step 1.8: Add children");
            // Add all of its children to the tree with empty node data, and track their parent
            //  As is, a node with multiple parents will be overwritten?
            self.children.insert(leaf.clone(), Vec::new());
            for neighbor in leaf.neighbors(mgr) {
                //TODO: self.t doesn't accurately count states present in self.children
                //TODO: self.children must be a DAG or select_leaf may not terminate
                self.children.get_mut(&leaf).expect("Leaf not found!").push(neighbor.clone());
                // Every node has a unique parent, so only update parents for newly discovered nodes
                if !((self.visited).contains(&neighbor)) {
                    self.visited.insert(neighbor.clone());
                    self.parents.insert(neighbor.clone(), leaf.clone());
                    self.t += 1;
                    println!("{}", self.t);
                    println!("{}", neighbor == leaf);
                    self.state_data.insert(neighbor.clone(), StateData::new());
                }
            }
            println!("Step 3: Roll out");
            println!("Root still no parents: {}", self.parents.get(&self.initial_state).is_none());
            // Step 2, Roll out from the leaf
            //TODO: should this roll out from one of the children?
            // Step 3, propagate the rollout score up the tree
            self.rollout_backprop(&leaf, rollout_length, mgr, rng, &heuristic);
        }
    }

    // Returns a reference to the best state found so far
    pub fn best_so_far(&self) -> Option<&T> {
        return None //TODO
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
            for neighbor in neighbors {
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
}
