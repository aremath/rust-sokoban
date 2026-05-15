use rand::prelude::*;
use rand::rngs::SmallRng;
use strum::IntoEnumIterator;

use crate::sokoengine::{Direction, SokoInterface};
use crate::sokoset::{SokoSet, SetManager, Pattern};
use crate::mcts::{Searchable, Tagged};

pub enum SetAction {
    ApplyPattern(Direction, Pattern),
    Collapse,
}

impl Searchable for SokoSet {
    type V = SetManager;
    type A = SetAction;

    fn neighbors(&self, mgr: &SetManager) -> Vec<(Self::A, SokoSet)> {
        let v = Vec::new();
        // Try to apply each pattern at each location
        for d in Direction::iter() {
            for p in &(mgr.patterns) {
                pv = vec![p];
                let state_next_o = self.state.apply_at_plocs(d, pv, mgr);
                match state_next_o {
                    Some(s) => {
                        let action = SetAction::ApplyPattern(d, p);
                        v.push((action, s)); },
                    None => {}
                }
            }
        }
        return v; //TODO
    }

    fn terminal(&self) -> bool {
        match self.state.resolve_singleton() {
            Some(_) => { true },
            None => { false }
        }
    }

    // TODO: you can't "win" an in-progress design (?)
    fn is_win(&self) -> bool {
        match self.state.resolve_singleton() {
            Some(s) => { SokoInterface::is_win(&s) },
            None => { false }
        }
    }
}

// Each SokoSearch needs a "mutable" reference to the underlying rng object
// But none of them will ever be mutable simultaneously
//TODO: I have no idea if this Rc<Refcell<T>> idea will work...
pub struct SokoSearch {
    state: Tagged<SokoSet>,
    rng: Rc<RefCell<SmallRng>>,
}


//TODO: I don't have MCTS set up to handle actions that might have nondeterministic outcomes...
//TODO: this needs an &mut self
impl Searchable for SokoSearch {
    type V = SetManager;
    type A = SetAction;

    fn neighbors(&self, mgr: &V) -> Vec<(Self::A, Self) {
        let v = Vec::new();
        let s_neighbs = self.state.neighbors();
        for (n_a, n_s) in s_neighbs {
            n_s_search = SokoSearch { state: n_s, rng: self.rng.clone() };
            v.push((n_a, n_s_search));
        }
        // Also consider random resolution
        //TODO: do we really have to uselessly clone this?
        let mut resolved = self.state.0.clone();
        {
            let mut rng = *(self.rng).borrow_mut();
            resolved = self.state.0.resolve_randomly(rng);
            // Let the mutable reference go out of scope
        }
        let resolved_s = SokoSearch { state: (resolved, self.state.1 + 1), rng: self.rng.clone() };
        v.push(resolved_s);
        return v;
    }
}
