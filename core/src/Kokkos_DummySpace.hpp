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

#ifndef KOKKOS_DUMMYDEVICE_HPP
#define KOKKOS_DUMMYDEVICE_HPP

#include <Kokkos_Macros.hpp>

#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_HostSpace.hpp>

/*--------------------------------------------------------------------------*/

namespace Kokkos {

/** \brief  Cuda on-device memory management */

class DummySpace {
public:

  //! Tag this class as a kokkos memory space
  typedef DummySpace             memory_space ;
  typedef Kokkos::Serial         execution_space ;
  typedef Kokkos::Device<execution_space,memory_space> device_type;

  typedef size_t                 size_type ;

  /*--------------------------------*/

  Kokkos::HostSpace host_space;

  /*--------------------------------*/

  DummySpace();
  DummySpace( DummySpace && rhs ) = default ;
  DummySpace( const DummySpace & rhs ) = default ;
  DummySpace & operator = ( DummySpace && rhs ) = default ;
  DummySpace & operator = ( const DummySpace & rhs ) = default ;
  ~DummySpace() = default ;

  /**\brief  Allocate untracked memory in the cuda space */
  void * allocate( const size_t arg_alloc_size ) const ;

  /**\brief  Deallocate untracked memory in the cuda space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const ;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

  /*--------------------------------*/
  /** \brief  Error reporting for HostSpace attempt to access DummySpace */
  static void access_error();

private:

  int  m_device ; ///< Which Cuda device

  static constexpr const char* m_name = "Dummy";
  friend class Kokkos::Impl::SharedAllocationRecord< Kokkos::DummySpace , void > ;
};

} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

static_assert( Kokkos::Impl::MemorySpaceAccess< Kokkos::DummySpace , Kokkos::DummySpace >::assignable , "" );

//----------------------------------------

template<>
struct MemorySpaceAccess< Kokkos::HostSpace , Kokkos::DummySpace > {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy   = true };
};

//----------------------------------------

template<>
struct MemorySpaceAccess< Kokkos::DummySpace , Kokkos::HostSpace > {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy   = true };
};

//----------------------------------------

}} // namespace Kokkos::Impl

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

template<typename ExecSpace> struct DeepCopy< DummySpace , DummySpace , ExecSpace>
{
  DeepCopy( void * dst , const void * src , size_t n ) {
    DeepCopy<HostSpace, HostSpace, ExecSpace>(dst, src, n);
  }
  DeepCopy( const ExecSpace& exec, void * dst , const void * src , size_t n) {
    DeepCopy<HostSpace, HostSpace, ExecSpace>(exec, dst, src, n);
  }
};

template<typename ExecSpace> struct DeepCopy< DummySpace , HostSpace , ExecSpace>
{
  DeepCopy( void * dst , const void * src , size_t n) {
    DeepCopy<HostSpace, HostSpace, ExecSpace>(dst, src, n);
  }
  DeepCopy( const Kokkos::DefaultHostExecutionSpace & exec, void * dst , const void * src , size_t n) {
    DeepCopy<HostSpace, HostSpace, ExecSpace>(exec, dst, src, n);
  }
};

template<typename ExecSpace> struct DeepCopy< HostSpace , DummySpace , ExecSpace>
{
  DeepCopy( void * dst , const void * src , size_t n) {
    DeepCopy<HostSpace, HostSpace, ExecSpace>(dst, src, n);
  }
  DeepCopy( const Kokkos::DefaultHostExecutionSpace & exec, void * dst , const void * src , size_t n) {
    DeepCopy<HostSpace, HostSpace, ExecSpace>(exec, dst, src, n);
  }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** Running in DummySpace attempting to access HostSpace: error */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::DummySpace , Kokkos::HostSpace >
{
  enum { value = false };
  KOKKOS_INLINE_FUNCTION static void verify( void )
    { Kokkos::abort("Dummy device code attempted to access HostSpace memory"); }

  KOKKOS_INLINE_FUNCTION static void verify( const void * )
    { Kokkos::abort("Dummy device code attempted to access HostSpace memory"); }
};

/** Running in DummySpace attempting to access an unknown space: error */
template< class OtherSpace >
struct VerifyExecutionCanAccessMemorySpace<
  typename enable_if< ! is_same<Kokkos::DummySpace,OtherSpace>::value , Kokkos::DummySpace >::type ,
  OtherSpace >
{
  enum { value = false };
  KOKKOS_INLINE_FUNCTION static void verify( void )
    { Kokkos::abort("Dummy device code attempted to access unknown Space memory"); }

  KOKKOS_INLINE_FUNCTION static void verify( const void * )
    { Kokkos::abort("Dummy device code attempted to access unknown Space memory"); }
};

//----------------------------------------------------------------------------
/** Running in HostSpace attempting to access DummySpace */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::HostSpace , Kokkos::DummySpace >
{
  enum { value = false };
  inline static void verify( void ) { DummySpace::access_error(); }
  inline static void verify( const void * ) { DummySpace::access_error(); }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template<>
class SharedAllocationRecord< Kokkos::DummySpace , void >
  : public SharedAllocationRecord< void , void >
{
private:

  typedef SharedAllocationRecord< void , void >  RecordBase ;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete ;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete ;

  static void deallocate( RecordBase * );

  static RecordBase s_root_record ;

  const Kokkos::DummySpace m_space ;

protected:

  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase(), m_space() {}

  SharedAllocationRecord( const Kokkos::DummySpace        & arg_space
                        , const std::string              & arg_label
                        , const size_t                     arg_alloc_size
                        , const RecordBase::function_type  arg_dealloc = & deallocate
                        );

public:

  inline
  std::string get_label() const
  {
    return std::string( RecordBase::head()->m_label );
  }

  static
  SharedAllocationRecord * allocate( const Kokkos::DummySpace &  arg_space
                                   , const std::string       &  arg_label
                                   , const size_t               arg_alloc_size
                                   )
  {
    return new SharedAllocationRecord( arg_space, arg_label, arg_alloc_size );
  }

  /**\brief  Allocate tracked memory in the space */
  static
  void * allocate_tracked( const Kokkos::DummySpace & arg_space
                         , const std::string & arg_label
                         , const size_t arg_alloc_size );

  /**\brief  Reallocate tracked memory in the space */
  static
  void * reallocate_tracked( void * const arg_alloc_ptr
                           , const size_t arg_alloc_size );

  /**\brief  Deallocate tracked memory in the space */
  static
  void deallocate_tracked( void * const arg_alloc_ptr );

  static SharedAllocationRecord * get_record( void * arg_alloc_ptr );

  static void print_records( std::ostream & , const Kokkos::DummySpace & , bool detail = false );
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif

