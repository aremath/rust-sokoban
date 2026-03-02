use std::collections::HashMap;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::cmp::Ordering;
use std::hash::Hash;
use ordered_float::OrderedFloat;
use strum::IntoEnumIterator;
use std::collections::VecDeque;

use crate::sokoengine::{SokoManager, SokoState, HasVecs, MapTile, Entity, Direction, SokoInterface, Stringable};

pub trait Searchable {
    type V: HasVecs;
    fn neighbors(&self, mgr: &Self::V) -> Vec<Self> where Self: Sized;
}

impl Searchable for SokoState<MapTile, Entity> {
    type V = SokoManager<MapTile, Entity>;

    fn neighbors(&self, mgr: &SokoManager<MapTile, Entity>) -> Vec<SokoState<MapTile, Entity>> {
        let mut v = Vec::new();
        for d in Direction::iter() {
            match self.update(d, mgr) {
                Some(s) => { v.push(s) },
                None => {}
            }
        }
        return v;
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
    n_visits: usize, // Keeps track of the number of times we have seen this state during a rollout
    value_estimate: OrderedFloat<f64>,
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
    visited: Box<HashSet<T>>,
    state_tree: Box<HashMap<T, T>>,
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
        v.insert(initial_state.clone());
        let s0 = SearchState { p: OrderedFloat(0.0), state_id: 0, t: initial_state.clone() };
        p.push(s0);
        return SearchTree { initial_state: initial_state.clone(), t: 1, visited : Box::new(v), state_tree : Box::new(h), state_data: Box::new(d), state_queue : Box::new(p) };
    }


    pub fn n_states(&self) -> usize {
        return self.visited.len();
    }

    // BFS
    //TODO: Stop all the cloning!
    // Instead, have visited hold the master list of discovered states,
    // and make everything else use references
    // Does not use the standard self.state_queue, which is a priority minqueue
    pub fn basic_search(&mut self, until: Option<fn(&T) -> bool>, max_states: Option<usize>, mgr: &<T as Searchable>::V) -> Option<T> {
        let mut queue = VecDeque::new();
        queue.push_back(self.initial_state.clone());
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
                    self.state_tree.insert(neighbor.clone(), parent.clone());
                    //let n0 = SearchState { p: OrderedFloat(self.t as f64), state_id: self.t, t: neighbor.clone() };
                    self.t = self.t + 1;
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
            match self.state_tree.get(current_state) {
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
