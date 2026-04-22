// Extends the basic sokoban engine from sokoengine.rs to include a set representation of sokoban levels
// Includes rules for interacting with the set representation, and bitsets for representing sets of designs
// Compatible with the same string representation as sokoengine.rs
extern crate nalgebra as na;
use rand::prelude::*;
use rand::rngs::SmallRng;
use na::Vector2;
use bimap::BiMap;
use ndarray::{Array, Ix2};

use crate::sokoengine;
use crate::sokoengine::{MapTile, Entity, Direction, SokoInterface, SokoState};

const NUM_PATTERNS: usize = 2;

//TODO: some way to use a single u16 instead of tuples for (map, entity)?

pub type MapSet = u8;
pub type EntitySet = u8;

const MAP_BLANK: MapSet = 1;
const MAP_WALL: MapSet = 2;
const MAP_TARGET: MapSet = 4;
const MAP_ANY: MapSet = 7;
const MAP_NOWALL: MapSet = 5;

const ENTITY_BLANK: EntitySet = 1;
const ENTITY_BLOCK: EntitySet = 2;
const ENTITY_PLAYER: EntitySet = 4;
const ENTITY_ANY: EntitySet = 7;
const ENTITY_NOPLAYER: EntitySet = 3;

fn set_to_tile(m: MapSet) -> Option<MapTile> {
    match m {
        MAP_BLANK => { Some(MapTile::Blank) },
        MAP_WALL => { Some(MapTile::Wall) },
        MAP_TARGET => { Some(MapTile::Target) },
        // Anything else is a non-singleton set (or invalid)
        _ => { None }
    }
}

fn set_to_entity(e: EntitySet) -> Option<Entity> {
    match e {
        ENTITY_BLANK => { Some(Entity::Blank) },
        ENTITY_BLOCK => { Some(Entity::Block) },
        ENTITY_PLAYER => { Some(Entity::Player) },
        _ => { None }
    }
}

impl sokoengine::IsPlayer for EntitySet {
    fn is_player(&self) -> bool {
        return *self == ENTITY_PLAYER;
    }
}

//TODO: cannot implement Default for u8, but it should be fine. It's only really used for out of bounds access...

// Create the symbol table for sokoset
pub fn mk_set_to_text() -> BiMap<(MapSet, EntitySet), char> {
    let mut set_to_text = BiMap::new();
    set_to_text.insert((MAP_BLANK, ENTITY_BLANK),    '_');
    set_to_text.insert((MAP_BLANK, ENTITY_BLOCK),    'B');
    set_to_text.insert((MAP_BLANK, ENTITY_PLAYER),   'P');
    set_to_text.insert((MAP_WALL,  ENTITY_BLANK),    'W');
    set_to_text.insert((MAP_TARGET, ENTITY_BLANK),   'O');
    set_to_text.insert((MAP_TARGET, ENTITY_BLOCK),   'G');
    set_to_text.insert((MAP_TARGET, ENTITY_PLAYER),  'Q');
    return set_to_text;
}

pub struct SetManager {
    m: sokoengine::SokoManager<MapSet, EntitySet>,
    patterns: Patterns,
}

impl SetManager {

    pub fn new(m: sokoengine::SokoManager<MapSet, EntitySet>, patterns: Patterns) -> Self {
        return SetManager { m: m, patterns: patterns };
    }

}

impl sokoengine::Coder<MapSet, EntitySet> for SetManager {
    fn encode(&self, t: (MapSet, EntitySet)) -> char {
        match self.m.symbol_table.get_by_left(&t) {
            Some(&c) => c,
            None => '?' // For any tile which has multiple possibilities
        }
    }

    fn decode(&self, c: char) -> (MapSet, EntitySet) {
        return self.m.decode(c);
    }
}

impl sokoengine::HasVecs for SetManager {

    fn d_to_v(&self, d: Direction) -> &Vector2<isize> {
        return self.m.d_to_v(d);
    }

}

#[derive(Clone)]
pub struct PatternEntry {
    cond: (MapSet, EntitySet), // Preconditions that are required to apply the pattern
    post: (MapSet, EntitySet)  // Things that happen as a result of applying the pattern (0 if nothing)
}

type Pattern = Vec<PatternEntry>;
type Patterns = Vec<Pattern>;

pub fn basic_patterns() -> Patterns {
    let basic_move_a = PatternEntry { cond: (MAP_NOWALL, ENTITY_PLAYER), post: (0, ENTITY_BLANK) };
    let basic_move_b = PatternEntry { cond: (MAP_NOWALL, ENTITY_BLANK), post: (0, ENTITY_PLAYER) };
    let basic_move = vec![basic_move_a, basic_move_b];
    //
    let basic_push_a = PatternEntry { cond: (MAP_NOWALL, ENTITY_PLAYER), post: (0, ENTITY_BLANK) };
    let basic_push_b = PatternEntry { cond: (MAP_NOWALL, ENTITY_BLOCK), post: (0, ENTITY_PLAYER) };
    let basic_push_c = PatternEntry { cond: (MAP_NOWALL, ENTITY_BLANK), post: (0, ENTITY_BLOCK) };
    let basic_push = vec![basic_push_a, basic_push_b, basic_push_c];
    return vec![basic_move, basic_push];
}

pub fn gen_patterns() -> Patterns {
    let b_patterns = basic_patterns();
    // Movement blocked by a wall
    let move_blocked_a = PatternEntry { cond: (MAP_NOWALL, ENTITY_PLAYER), post: (0, ENTITY_PLAYER) };
    let move_blocked_b = PatternEntry { cond: (MAP_WALL, ENTITY_BLANK), post: (0, ENTITY_BLANK) };
    let move_blocked = vec![move_blocked_a, move_blocked_b];
    // Block push blocked by a wall
    let push_blocked_a = PatternEntry { cond: (MAP_NOWALL, ENTITY_PLAYER), post: (0, ENTITY_PLAYER) };
    let push_blocked_b = PatternEntry { cond: (MAP_NOWALL, ENTITY_BLOCK), post: (0, ENTITY_BLOCK) };
    let push_blocked_c = PatternEntry { cond: (MAP_WALL, ENTITY_BLANK), post: (0, ENTITY_BLANK) };
    let push_blocked = vec![push_blocked_a, push_blocked_b, push_blocked_c];
    // Block push blocked by another block
    let push_blocked_block_a = PatternEntry { cond: (MAP_NOWALL, ENTITY_PLAYER), post: (0, ENTITY_PLAYER) };
    let push_blocked_block_b = PatternEntry { cond: (MAP_NOWALL, ENTITY_BLOCK), post: (0, ENTITY_BLOCK) };
    let push_blocked_block_c = PatternEntry { cond: (MAP_NOWALL, ENTITY_BLOCK), post: (0, ENTITY_BLOCK) };
    let push_blocked_block = vec![push_blocked_block_a, push_blocked_block_b, push_blocked_block_c];
    let mut patterns = Vec::new();
    patterns.extend(b_patterns);
    patterns.extend(vec![move_blocked, push_blocked, push_blocked_block]);
    return patterns;
}

pub type SokoSet = SokoState<MapSet, EntitySet>;

impl SokoSet {
    
    // Updates self with the given pattern, returning whether the pattern succeeded
    // This allows multiple patterns to be applied without a clone()
    fn mut_pattern_at<V: sokoengine::HasVecs>(&mut self, c: &Vector2<isize>, d: Direction,
        pattern: &[PatternEntry], mgr: &V) -> bool {
        let mut current_pos = c.clone();
        let dv = mgr.d_to_v(d);
        // Check if the preconditions are met
        for p in pattern {
            let m = self.get_tile(&current_pos);
            let e = self.get_entity(&current_pos);
            // The rule fails if either possibility set is empty
            if (m & p.cond.0) == 0 || (e & p.cond.1) == 0 {
                return false;
            }
            current_pos = current_pos + dv;
        }
        // If the preconditions are met, then apply them.
        // Also apply any necessary postconditions
        current_pos = c.clone();
        for p in pattern {
            if p.post.0 != 0 {
                self.set_tile(&current_pos, p.post.0);
            } else {
                let m = self.get_tile(&current_pos);
                self.set_tile(&current_pos, m & p.cond.0);
            }
            if p.post.1 != 0 {
                self.set_entity(&current_pos, p.post.1);
            } else {
                let e = self.get_entity(&current_pos);
                self.set_entity(&current_pos, e & p.cond.1);
            }
            current_pos = current_pos + dv;
        }
        return true;
    }

    // Try to apply a pattern at a specific location. Other functions will determine where to apply them
    fn apply_pattern_at<V: sokoengine::HasVecs>(&self, c: &Vector2<isize>, d: Direction,
        pattern: &[PatternEntry], mgr: &V) -> Option<SokoSet> {
        let mut new_state = self.clone();
        let mut current_pos = c.clone();
        let dv = mgr.d_to_v(d);
        // Check if the preconditions are met
        for p in pattern {
            let m = self.get_tile(&current_pos);
            let e = self.get_entity(&current_pos);
            if (m & p.cond.0) == 0 || (e & p.cond.1) == 0 {
                return None;
            }
            current_pos = current_pos + dv;
        }
        // If the preconditions are met, apply the postconditions
        current_pos = c.clone();
        for p in pattern {
            if p.post.0 != 0 {
                new_state.set_tile(&current_pos, p.post.0);
            } else {
                let m = new_state.get_tile(&current_pos);
                new_state.set_tile(&current_pos, m & p.cond.0);
            }
            if p.post.1 != 0 {
                new_state.set_entity(&current_pos, p.post.1);
            } else {
                let e = new_state.get_entity(&current_pos);
                new_state.set_entity(&current_pos, e & p.cond.1);
            }
            current_pos = current_pos + dv;
        }
        return Some(new_state);
    }

    //TODO: do I need a version of this that applies multiple patterns mutably?
    fn apply_at_plocs<V: sokoengine::HasVecs>(&self, d: Direction,
        patterns: &Patterns, mgr: &V) -> Option<SokoSet> {
        //println!("OLD {:?}", self.player_locs);
        let ordering = sokoengine::choose_ordering(d);
        let dv = mgr.d_to_v(d);
        let mut locs = self.player_locs.to_vec();
        let mut new_state = self.clone();
        let mut changed = false;
        let mut new_plocs: Vec<Vector2<isize>> = Vec::new();
        locs.sort_by_key(ordering);
        for loc in locs {
            let mut loc_changed = false;
            for pattern in patterns {
                let b = new_state.mut_pattern_at(&loc, d, pattern, mgr);
                // If any pattern succeeds, then the state is changed*
                changed = changed || b;
                loc_changed = loc_changed || b;
                if b {
                    let loc_next = loc + dv;
                    // Not all patterns will actually change the player position when they succeed
                    if new_state.get_entity(&loc_next) == ENTITY_PLAYER {
                        new_plocs.push(loc_next);
                    }
                    // At most one pattern can succeed per step
                    break;
                }
            }
            // If no patterns were applied, we still need to retain the location
            if !loc_changed {
                //println!("Not Changed!");
                new_plocs.push(loc);
            }
        }
        if changed {
            //println!("NEW {:?}", new_plocs);
            new_state.player_locs = new_plocs;
            return Some(new_state);
        } else {
            return None;
        }
    }

    // Resolve a SokoSet to a singleton SokoState
    // Returns None if the SokoSet is not singleton
    pub fn resolve_singleton(&self) -> Option<SokoState<MapTile, Entity>> {
        let shape = self.map_layer.shape();
        let height = shape[0];
        let width = shape[1];
        let mut locs = Vec::<Vector2<isize>>::new();
        let mut tiles = Array::<MapTile, Ix2>::default((height, width));
        let mut entities = Array::<Entity, Ix2>::default((height, width));
        for (i, _) in self.map_layer.indexed_iter() {
            let v = Vector2::new(i.0 as isize, i.1 as isize);
            let t_set = self.get_tile(&v);
            let o_m = set_to_tile(t_set);
            match o_m {
                Some(m) => { tiles[i] = m; },
                None => { return None; }
            }
            let e_set = self.get_entity(&v);
            let o_e = set_to_entity(e_set);
            match o_e {
                Some(e) => { entities[i] = e;
                    if e == Entity::Player {
                        locs.push(v);
                    }
                },
                None => { return None; }
            }
        }
        return Some(SokoState { map_layer: tiles, entity_layer: entities, player_locs: locs });
    }

    fn resolve_randomly(&self, rng: &mut SmallRng) -> SokoSet {
        let tile_matches = vec![MAP_BLANK, MAP_WALL, MAP_TARGET];
        let entity_matches = vec![ENTITY_BLANK, ENTITY_BLOCK, ENTITY_PLAYER];
        let shape = self.map_layer.shape();
        let height = shape[0];
        let width = shape[1];
        let mut locs = Vec::<Vector2<isize>>::new();
        let mut tiles = Array::<MapSet, Ix2>::default((height, width));
        let mut entities = Array::<EntitySet, Ix2>::default((height, width));
        for (i, _) in self.map_layer.indexed_iter() {
            let v = Vector2::new(i.0 as isize, i.1 as isize);
            let t_set = self.get_tile(&v);
            let t_matches: Vec<MapSet> = tile_matches.clone().into_iter().filter(|x| (x & t_set) != 0).collect();
            let t_match = t_matches.choose(rng).expect("No matches!");
            tiles[i] = *t_match;
            let e_set = self.get_entity(&v);
            let e_matches: Vec<EntitySet> = entity_matches.clone().into_iter().filter(|x| (x & e_set) != 0).collect();
            let e_match = e_matches.choose(rng).expect("No matches!");
            entities[i] = *e_match;
            if *e_match == ENTITY_PLAYER {
                locs.push(v);
            }
        }
        return SokoSet { map_layer: tiles, entity_layer: entities, player_locs: locs };
    }
}

impl SokoInterface<MapSet, EntitySet> for SokoSet {
    type V = SetManager;

    fn update(&self, d: Direction, mgr: &SetManager) -> Option<Self> {
        return self.apply_at_plocs(d, &mgr.patterns, mgr);
    }

    //TODO: could consider some definitions of victory such as:
    //  1. All possible configurations win
    //  2. There exists a winning configuration
    //  Both of these seem problematic
    fn is_win(&self) -> bool {
        for (i, _) in self.map_layer.indexed_iter() {
            if self.map_layer[i] == MAP_TARGET && !(self.entity_layer[i] == ENTITY_BLOCK) {
                return false;
            }
        }
        return true;
    }
}
