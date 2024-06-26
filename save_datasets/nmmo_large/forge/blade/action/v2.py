from pdb import set_trace as T
from forge.blade.action import action
from forge.blade.lib import utils, enums
import numpy as np
import random
class Arg:
   def __init__(self, val, discrete=True, set=False, min=-1, max=1):
      self.val = val
      self.discrete = discrete
      self.continuous = not discrete
      self.min = min
      self.max = max
      self.n = self.max - self.min + 1


class FishingEnvironment:
   def __init__(self):
      self.fish_population = 100  # 初始鱼群数量

   def reduce_fish_population(self, amount):
      """减少鱼群数量，模拟捕鱼动作的结果"""
      self.fish_population = max(0, self.fish_population - amount)
      return amount

   def get_fish_population(self):
      """获取当前鱼群的数量"""
      return self.fish_population



class ActionV2:
   def edges(self, world, entity, inclusive=False):
      return [Pass, Move, Attack]#, Ranged]

class Pass(action.Pass):
   priority = 0

   @staticmethod
   def call(world, entity):
      return

   def args(stim, entity, config):
      return [()]

   @property
   def nArgs():
      return 1

class Move(action.Move):
   priority = 1
   def call(world, entity, rDelta, cDelta):
      r, c = entity.pos
      rNew, cNew = r+rDelta, c+cDelta
      if world.env.tiles[rNew, cNew].state.index in enums.IMPASSIBLE:
         return
      if not utils.inBounds(rNew, cNew, world.shape):
         return
      if entity.freeze > 0:
         return

      entity._pos = rNew, cNew
      entID = entity.entID
      
      r, c = entity.lastPos
      world.env.tiles[r, c].delEnt(entID)

      r, c = entity.pos
      world.env.tiles[r, c].addEnt(entID, entity)

   def args(stim, entity, config):
      rets = []
      for delta in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)):
         r, c = delta
         #r, c = Arg(r), Arg(c)
         rets.append((r, c))
      return rets

   @property
   def nArgs():
      return len(Move.args(None, None))

class Attack(action.Attack):
   def inRange(entity, stim, N):
      R, C = stim.shape
      R, C = R//2, C//2
      #R, C = entity.pos

      rets = []
      for r in range(R-N, R+N+1):
         for c in range(C-N, C+N+1):
            for e in stim[r, c].ents.values():
               rets.append(e)
      return rets

   def l1(pos, cent):
      r, c = pos
      rCent, cCent = cent
      return abs(r - rCent) + abs(c - cCent)

   def call(world, entity, targ, damageF, freeze=False):
      if entity.entID == targ.entID:
         entity._attack = None
         return
      #entity.targPos = targ.pos
      #entity.attkPos = entity.lastPos
      #entity.targ = targ
      damage = damageF(entity, targ)
      assert type(damage) == int
      if freeze and damage > 0:
         targ._freeze = 3
      return
      #return damage

   def args(stim, entity, config):
      return [Melee, Range, Mage]
      #return Melee.args(stim, entity, config) + Range.args(stim, entity, config) + Mage.args(stim, entity, config)

class Melee(action.Melee):
   priority = 2
   def call(world, entity, targ):
      damageF = world.config.MELEEDAMAGE
      Attack.call(world, entity, targ, damageF)

   def args(stim, entity, config):
      return Attack.inRange(entity, stim, config.MELEERANGE)

class Range(action.Range):
   priority = 2
   def call(world, entity, targ):
      damageF = world.config.RANGEDAMAGE
      Attack.call(world, entity, targ, damageF)

   def args(stim, entity, config):
      return Attack.inRange(entity, stim, config.RANGERANGE)

class Mage(action.Mage):
   priority = 2
   def call(world, entity, targ):
      damageF = world.config.MAGEDAMAGE
      dmg = Attack.call(world, entity, targ, damageF, freeze=True)

   def args(stim, entity, config):
      return Attack.inRange(entity, stim, config.MAGERANGE)


class Farming(action.Farming):
   priority = 3
   def call(world, entity):
      r, c = entity.pos
      if world.env.tiles[r, c].state.index in enums.IMPASSIBLE:
         cultivated = 1
         print(f"Planted at ({r}, {c})")
         return cultivated
      else:
         print("Attempted to plant outside of farm boundaries.")

class Fish(action.Fish):
   priority = 4
   def call(world):
      if world.get_fish_population() > 0:
         fish_caught = random.randint(1, 10)
         fish_caught = world.env.reduce_fish_population(fish_caught)
         print(f"Caught {fish_caught} fish. Total fish caught: {fish_caught}")
         return fish_caught
      else:
         print("No more fish left to catch.")

class Climbing(action.Climbing):
   priority = 5
   def call(world, entity, rDelta, cDelta):

      r, c = entity.pos
      rNew, cNew = r+rDelta, c+cDelta
      if world.env.tiles[rNew, cNew].state.index in enums.IMPASSIBLE:
         return
      if not utils.inBounds(rNew, cNew, world.shape):
         return
      if entity.freeze > 0:
         return

      entity._pos = rNew, cNew
      entID = entity.entID

      r, c = entity.lastPos
      world.env.tiles[r, c].delEnt(entID)

      r, c = entity.pos
      world.env.tiles[r, c].addEnt(entID, entity)