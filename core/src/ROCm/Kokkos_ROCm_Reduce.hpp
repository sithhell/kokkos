/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

///////////////////////////////////////////////////////////////////////////////
// AMP REDUCE
//////////////////////////////////////////////////////////////////////////////

#if !defined( KOKKOS_ROCM_AMP_REDUCE_INL )
#define KOKKOS_ROCM_AMP_REDUCE_INL

#include <iostream>

#include <algorithm>
#include <numeric>
#include <cmath>
#include <type_traits>
#include <ROCm/Kokkos_ROCm_Tile.hpp>
#include <ROCm/Kokkos_ROCm_Invoke.hpp>
#include <ROCm/Kokkos_ROCm_Join.hpp>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Kokkos {
namespace Impl {

template<class T>
T* reduce_value(T* x, std::true_type) [[hc]]
{
  return x;
}

template<class T>
T& reduce_value(T* x, std::false_type) [[hc]]
{
  return *x;
}

#if KOKKOS_ROCM_HAS_WORKAROUNDS
struct always_true
{
    template<class... Ts>
    bool operator()(Ts&&...) const
    {
        return true;
    }
};
#endif

template< class Tag, class F, class ReducerType, class Invoker, class T >
void reduce_enqueue(
  const int szElements,  // size of the extent
  const F & f,
  const ReducerType& reducer,
  Invoker invoke,
  T * const output_result,
  int const output_length,
  const int team_size=64,
  const int vector_size=1,
  int const shared_size=0)
{
  using namespace hc ;

  typedef Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, F, ReducerType> ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;

  typedef Kokkos::Impl::FunctorValueTraits< ReducerTypeFwd , Tag > ValueTraits ;
  typedef Kokkos::Impl::FunctorValueInit< ReducerTypeFwd , Tag >   ValueInit ;
  typedef Kokkos::Impl::FunctorValueJoin< ReducerTypeFwd , Tag >   ValueJoin ;
  typedef Kokkos::Impl::FunctorFinal< ReducerTypeFwd , Tag >       ValueFinal ;

  typedef typename ValueTraits::pointer_type   pointer_type ;
  typedef typename ValueTraits::reference_type reference_type ;

  if (output_length < 1) return;

  const auto td = get_tile_desc<T>(szElements,output_length,team_size,vector_size, shared_size);

  // allocate host and device memory for the results from each team
  std::vector<T> result_cpu(td.num_tiles*output_length);
  auto result = reinterpret_cast<T*>(
          hc2::accelerator{}.get_dedicated_memory_allocator().allocate(
          sizeof(T) * td.num_tiles * output_length));

  auto fut = tile_for<T[]>(td, [=](hc::tiled_index<1> t_idx, tile_buffer<T[]> buffer) [[hc]] 
  {
      const auto local = t_idx.local[0];
      const auto global = t_idx.global[0];
      const auto tile = t_idx.tile[0];

      buffer.action_at(local, [&](T* state)
      {
          ValueInit::init(ReducerConditional::select(f, reducer), state);
          invoke(make_rocm_invoke_fn<Tag>(f), t_idx, td, reduce_value(state, std::is_pointer<reference_type>()));
      });
      t_idx.barrier.wait();

      // Reduce within a tile using multiple threads.
// even though buffer.size is always 64, the value 64 must be hard coded below
// due to a compiler bug
//      for(std::size_t s = 1; s < buffer.size(); s *= 2)
      for(std::size_t s = 1; s < 64; s *= 2)
      {
          const std::size_t index = 2 * s * local;
//          if (index < buffer.size())
          if (index < 64)
          {
              buffer.action_at(index, index + s, [&](T* x, T* y)
              {
                  ValueJoin::join(ReducerConditional::select(f, reducer), x, y);
              });
          }
          t_idx.barrier.wait();
      }

      // Store the tile result in the global memory.
      if (local == 0)
      {
          // Workaround for assigning from LDS memory: std::copy should work
          // directly
          buffer.action_at(0, [&](T* x)
          {
              for (auto i = 0u; i != output_length; ++i) {
                  result[tile * output_length + i] = x[i];
              }
          });
      }
      
  });
  if (output_result != nullptr)
     ValueInit::init(ReducerConditional::select(f, reducer), output_result);
  fut.wait();

  hc2::accelerator::copy(
          result, result + td.num_tiles * output_length, result_cpu.begin());
  if (output_result != nullptr) {
    for(std::size_t i=0;i<td.num_tiles;i++)
       ValueJoin::join(ReducerConditional::select(f, reducer), output_result, result_cpu.data()+i*output_length);

    ValueFinal::final( ReducerConditional::select(f, reducer) , output_result );
  }

}

}} //end of namespace Kokkos::Impl

#endif /* #if !defined( KOKKOS_ROCM_AMP_REDUCE_INL ) */

