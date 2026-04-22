// Basic Sokoban Engine
// Includes string input / output for sokoban levels, movement model for sokoban, etc.
// Also includes a memory type for implementing undo operations
use ndarray::{Array, Ix2};
use bimap::BiMap;
use std::hash::{Hash, Hasher};
extern crate approx;
extern crate nalgebra as na;
use na::Vector2;
use std::collections::VecDeque;
use std::marker::PhantomData;
use strum_macros::EnumIter;

const MAX_MEMORY: usize = 100;

#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
pub enum MapTile {
    Blank,
    Wall,
    Target,
}

impl MapTile {
    fn is_open(&self) -> bool {
        match self {
            MapTile::Blank => true,
            MapTile::Target => true,
            _ => false,
        }
    }
}

impl Default for MapTile {
    fn default() -> MapTile {
        return MapTile::Wall;
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
pub enum Entity {
    Blank,
    Block,
    Player,
}

impl Default for Entity {
    fn default() -> Entity {
        return Entity::Blank;
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq, EnumIter)]
pub enum Direction {
    Up,
    Left,
    Right,
    Down,
}

pub trait Coder<M, E> {
    fn encode(&self, t: (M, E)) -> char;
    fn decode(&self, c: char) -> (M, E);
}


// Trait for a type that can use a manager to read and write state to string
// This trait is here because the reading / writing code can be generic
// Stringable depends on C because different implementations of C can provide different resulting strings, even for the same M, E types.
pub trait Stringable<M, E, C> {
    fn from_str(s: &String, mgr: &C) -> Self;
    fn to_str(&self, mgr: &C) -> String;
}

pub trait HasVecs {
    fn d_to_v(&self, d: Direction) -> &Vector2<isize>;
}

// Trait for a way to check whether a thing counts as a player location
// Necessary for from_str to be polymorphic, because the player_locs list needs to be instantiated from the string
pub trait IsPlayer {
    fn is_player(&self) -> bool;
}

impl IsPlayer for Entity {
    fn is_player(&self) -> bool {
        return *self == Entity::Player;
    }
}

pub struct SokoManager<M, E> {
    pub symbol_table: BiMap<(M, E), char>,
    //symbol_table: BiMap<(MapTile, Entity), char>,
    v_up: Vector2<isize>,
    v_down: Vector2<isize>,
    v_left: Vector2<isize>,
    v_right: Vector2<isize>,
}

impl<M: Eq + Hash + Copy, E: Eq + Hash + Copy> SokoManager<M, E> {

    pub fn new(f: fn() -> BiMap<(M, E), char>) -> Self {
        let v_up = Vector2::new(-1, 0);
        let v_left = Vector2::new(0, -1);
        let v_right = Vector2::new(0, 1);
        let v_down = Vector2::new(1, 0);
        Self { symbol_table: f(), v_up, v_down, v_left, v_right }
    }
}

impl<M, E> HasVecs for SokoManager<M, E> {

    fn d_to_v(&self, d: Direction) -> &Vector2<isize> {
        match d {
            Direction::Up => &self.v_up,
            Direction::Left => &self.v_left,
            Direction::Right => &self.v_right,
            Direction::Down => &self.v_down,
        }
    }

}

impl<M: Eq + Hash + Copy, E: Eq + Hash + Copy> Coder<M, E> for SokoManager<M, E>
{
    fn encode(&self, t: (M, E)) -> char {
        return *self.symbol_table.get_by_left(&t).expect("Bad map, entity combination!");
    }

    fn decode(&self, c: char) -> (M, E) {
        return *self.symbol_table.get_by_right(&c).expect("Bad character!");
    }
}


//TODO: why do these fields have to be pub in order to use them in the sokoset type alias?
#[derive(Clone)]
pub struct SokoState<M, E> {
    pub map_layer : Array<M, Ix2>,
    pub entity_layer : Array<E, Ix2>,
    pub player_locs: Vec<Vector2<isize>>,   // Should never be negative, but have to update with isizes
}

// Need to implement Hash rather than derive it because of the type_to_text reference
// Two sokostates with the same entity_layer should always have the same player_locs (modulo reordering)
// Sokostates with a different player_locs order are still equivalent
// Two sokostates with different type_to_texts but the same actual data are equivalent
impl<M: Hash, E: Hash> Hash for SokoState<M, E> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.map_layer.hash(state);
        self.entity_layer.hash(state);
    }
}

impl<M: PartialEq, E: PartialEq> PartialEq for SokoState<M, E> {
    fn eq(&self, other: &Self) -> bool {
        self.map_layer == other.map_layer && self.entity_layer == other.entity_layer
    }

}

impl<M: Eq, E: Eq> Eq for SokoState<M, E> {}

// Consistent with Seth Cooper's sokoban
pub fn mk_type_to_text() -> BiMap<(MapTile, Entity), char> {
    let mut type_to_text = BiMap::new();
    type_to_text.insert((MapTile::Blank, Entity::Blank),    '_');
    type_to_text.insert((MapTile::Blank, Entity::Block),    'B');
    type_to_text.insert((MapTile::Blank, Entity::Player),   'P');
    type_to_text.insert((MapTile::Wall,  Entity::Blank),    'W');
    type_to_text.insert((MapTile::Target, Entity::Blank),   'O');
    type_to_text.insert((MapTile::Target, Entity::Block),   'G');
    type_to_text.insert((MapTile::Target, Entity::Player),  'Q');
    return type_to_text;
}

// Indexing is y,x because that's how rust ndarray rolls
fn y_increasing(a: &Vector2<isize>) -> isize {
    a.y
}

fn y_decreasing(a: &Vector2<isize>) -> isize {
    -a.y
}

fn x_increasing(a: &Vector2<isize>) -> isize {
    a.x
}

fn x_decreasing(a: &Vector2<isize>) -> isize {
    -a.x
}

// Choose the ordering so that multiple player entities moving in direction d do not block each other
pub fn choose_ordering(d: Direction) -> fn(&Vector2<isize>) -> isize {
    match d {
        Direction::Right => y_decreasing,  // +y
        Direction::Left => y_increasing,   // -y
        Direction::Down => x_decreasing,   // +x
        Direction::Up => x_increasing,     // -x
    }
}

pub fn index_checked<T>(a: &Array<T, Ix2>, c: &Vector2<isize>) -> Option<T>
    where T: Clone {
    return a.get((c.x as usize, c.y as usize)).cloned();
}

// All of the things expected of a Sokoban implementation
// Use the type syntax from (for example) IntoIterator to define V inside the trait!
// Any SokoInterface implementation needs to specify the Manager type
pub trait SokoInterface<M, E> {
    type V: HasVecs;
    fn update(&self, _: Direction, mgr: &Self::V) -> Option<Self> where Self: Sized;
    fn is_win(&self) -> bool;
}

impl<M: Default + Clone, E: Default + Clone> SokoState<M, E> {

    pub fn get_tile_maybe(&self, c: &Vector2<isize>) -> Option<M> {
        return index_checked(&self.map_layer, c);
    }

    pub fn get_entity_maybe(&self, c: &Vector2<isize>) -> Option<E> {
        return index_checked(&self.entity_layer, c);
    }
    
    pub fn get_tile(&self, c: &Vector2<isize>) -> M {
        match index_checked(&self.map_layer, c) {
            Some(a) => a,
            None => M::default(),
        }
    }

    pub fn get_entity(&self, c: &Vector2<isize>) -> E {
        match index_checked(&self.entity_layer, c) {
            Some(a) => a,
            None => E::default(),
        }
    }

    //TODO: why do these have to be pub in order to use them in the type alias in sokoset.rs?
    pub fn set_tile(&mut self, c: &Vector2<isize>, t: M) -> () {
        self.map_layer[(c.x as usize, c.y as usize)] = t;
    }

    pub fn set_entity(&mut self, c: &Vector2<isize>, e: E) -> () {
        self.entity_layer[(c.x as usize, c.y as usize)] = e;
    }
}

impl<M: Eq + Hash + Copy + Default, E: Eq + Hash + Copy + IsPlayer + Default, C: Coder<M, E>> Stringable<M, E, C> for SokoState<M, E> {

    fn to_str(&self, mgr: &C) -> String {
        let mut s = "".to_string();
        let mut cy = 0;
        for (i, _) in self.map_layer.indexed_iter() {
            // Insert newlines!
            if i.0 > cy {
                s.push('\n');
                cy = i.0;
            }
            // Here I am assuming that the map layer and entity layer have the same shape.
            let (m, e) = (self.map_layer[i], self.entity_layer[i]);
            let c = mgr.encode((m, e));
            s.push(c);
        }
        //write!(f, "{}", s)
        return s;
    }

    fn from_str(s: &String, mgr: &C) -> SokoState<M, E> {
        let width = s.lines().next().unwrap().len();
        let height = s.lines().count();
        let mut locs = Vec::<Vector2<isize>>::new();
        let mut tiles = Array::<M, Ix2>::default((height, width));
        let mut entities = Array::<E, Ix2>::default((height, width));
        for (i, line) in s.lines().enumerate() {
            for (j, c) in line.chars().enumerate() {
                let (m, e) = mgr.decode(c);
                tiles[(i, j)] = m;
                entities[(i, j)] = e;
                if e.is_player() {
                    locs.push(Vector2::new(i as isize, j as isize));
                }
            }
        }
        return SokoState{map_layer: tiles, entity_layer: entities, player_locs: locs};
    }

}

impl SokoInterface<MapTile, Entity> for SokoState<MapTile, Entity> {
    type V = SokoManager<MapTile, Entity>;

    fn update(&self, d: Direction, mgr: &Self::V) -> Option<SokoState<MapTile, Entity>> {
        //println!("{:?}", self.player_locs);
        let ordering = choose_ordering(d);
        let mut locs = self.player_locs.to_vec();
        locs.sort_by_key(ordering);
        let mut new_state = self.clone();
        let dv = mgr.d_to_v(d);
        //println!("{}", dv);
        let mut changed = false;
        let mut new_plocs: Vec<Vector2<isize>> = Vec::new();
        for loc in locs.iter() {
            // Try to move each player entity
            let loc_next = *loc + dv;
            // The proposed move must go into empty space
            // use new_state because some entities may have already moved
            if new_state.get_tile(&loc_next).is_open() {
                // Moving without pushing a box
                if new_state.get_entity(&loc_next) == Entity::Blank {
                    //println!("Moving from {} to {}", loc, loc_next);
                    new_state.set_entity(&loc, Entity::Blank);
                    new_state.set_entity(&loc_next, Entity::Player);
                    new_plocs.push(loc_next);
                    changed = true;
                // Pushing a box, possibly
                } else if new_state.get_entity(&loc_next) == Entity::Block {
                    let loc_next_next = loc_next + dv;
                    // If the target square is also open, then the box can be moved
                    if new_state.get_tile(&loc_next_next).is_open() && new_state.get_entity(&loc_next_next) == Entity::Blank {
                        //println!("Moving from {} to {}", loc, loc_next);
                        new_state.set_entity(&loc, Entity::Blank);
                        new_state.set_entity(&loc_next, Entity::Player);
                        new_state.set_entity(&loc_next_next, Entity::Block);
                        new_plocs.push(loc_next);
                        changed = true;
                    } else {
                        new_plocs.push(*loc);
                    }
                } else {
                    new_plocs.push(*loc);
                }
            } else {
                new_plocs.push(*loc);
                //println!("Proposed move is blocked by a wall");
            }
        }
        if changed {
            new_state.player_locs = new_plocs;
            //println!("{:?}", new_state.player_locs);
            return Some(new_state) // placeholder
        } else {
            return None
        }
    }

    fn is_win(&self) -> bool {
        //TODO: this could be sped up by changing it to an Array[(MapTile, Entity)]
        for (i, _) in self.map_layer.indexed_iter() {
            if self.map_layer[i] == MapTile::Target && !(self.entity_layer[i] == Entity::Block) {
                return false;
            }
        }
        return true;
    }
}

pub struct SokoMemory<M, E, C> {
    pub current_state: SokoState<M, E>,
    past_states: VecDeque<SokoState<M, E>>,
    phantom: PhantomData<C>, // Sokomemory doesn't HAVE a manager, but it depends on the type of manager being used
}

impl<M: Eq + Hash + Copy + Default, E: Eq + Hash + Copy + IsPlayer + Default, C: Coder<M, E>> Stringable<M, E, C> for SokoMemory<M, E, C> where
    SokoState<M, E>: Stringable<M, E, C>,
{

    fn to_str(&self, mgr: &C) -> String {
        self.current_state.to_str(mgr)
    }

    fn from_str(s: &String, mgr: &C) -> SokoMemory<M, E, C> {
        let s = SokoState::<M, E>::from_str(s, mgr);
        return SokoMemory { current_state: s, past_states: VecDeque::new(), phantom: PhantomData};
    }

}

//TODO: impl<T> SokoInterface<A> for SokoMemory<T>
// The main barrier here is that SokoInterface.update requires &self, but SokoMemory uses &mut self
// Either transition SokoMemory.update to use &self (and change past_states to VecDeque<&T>), or require &mut self
// The first is probably better, but would require a lifetime annotation for SokoMemory
//impl<M: SokoInterface<> + Clone + Stringable<A>> SokoMemory<A, T> {
impl<M: Clone, E: Clone, C: Coder<M, E>> SokoMemory<M, E, C> where 
    //C: <SokoState<M, E> as SokoInterface<M, E>>::V,
    SokoState<M, E>: SokoInterface<M, E>,
    SokoState<M, E>: Stringable<M, E, C>,
{

    pub fn update(&mut self, d: Direction, mgr: &<SokoState<M, E> as SokoInterface<M, E>>::V) -> () {
        let s_opt = self.current_state.update(d, mgr);
        match s_opt {
            // Only update the queue and the current state if the move succeeds
            Some(s_next) => {
                if self.past_states.len() >= MAX_MEMORY {
                    self.past_states.pop_front();
                }
                //TODO: there is probably some way to do this without cloning
                self.past_states.push_back(self.current_state.clone());
                self.current_state = s_next;
            }
            None => {}
        }
    }

    pub fn undo(&mut self) -> () {
        match self.past_states.pop_back()
        {
            Some(s) => self.current_state = s,
            None => println!("Cannot undo past queue end")
        }
    }

    pub fn is_win(&self) -> bool {
        self.current_state.is_win()
    }
}

