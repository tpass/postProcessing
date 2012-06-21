//@HEADER
/*
 *******************************************************************************

    Copyright (C) 2004, 2005, 2007 EPFL, Politecnico di Milano, INRIA
    Copyright (C) 2010 EPFL, Politecnico di Milano, Emory University

    This file is part of LifeV.

    LifeV is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    LifeV is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with LifeV.  If not, see <http://www.gnu.org/licenses/>.

 *******************************************************************************
 */
//@HEADER

/*!
    @file
    @brief

    @contributor Tiziano Passerini <tiziano@mathcs.emory.edu>
    @date 08-10-2010
 */

// ===================================================
//! Includes
// ===================================================
// Tell the compiler to ignore specific kind of warnings:
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <Epetra_ConfigDefs.h>
#ifdef EPETRA_MPI
#include <mpi.h>
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif

//Tell the compiler to restore the warning previously silented
#pragma GCC diagnostic warning "-Wunused-variable"
#pragma GCC diagnostic warning "-Wunused-parameter"

#include <lifev/core/util/Displayer.hpp>

#include <lifev/core/filter/ExporterEnsight.hpp>
#include <lifev/core/filter/ExporterHDF5.hpp>
#include <lifev/core/filter/ExporterEmpty.hpp>
#include <lifev/core/filter/ExporterVTK.hpp>

#include <lifev/core/mesh/MeshData.hpp>

#include <lifev/core/function/RossEthierSteinmanDec.hpp>

#include <lifev/core/solver/GradientCalculator.hpp>
#include <lifev/core/fem/PostProcessingBoundary.hpp>


using namespace LifeV;

typedef Epetra_Comm                                    comm_Type;
typedef boost::shared_ptr<comm_Type>                   commPtr_Type;
typedef LifeV::RegionMesh<LifeV::LinearTetra>          mesh_Type;
typedef boost::shared_ptr<mesh_Type>                   meshPtr_Type;
typedef LifeV::MatrixEpetra<LifeV::Real>               matrix_Type;
typedef LifeV::VectorEpetra                            vector_Type;
typedef boost::shared_ptr<vector_Type>                 vectorPtr_Type;
typedef LifeV::FESpace< mesh_Type, LifeV::MapEpetra >  feSpace_Type;
typedef boost::shared_ptr<feSpace_Type>                feSpacePtr_Type;
typedef LifeV::RossEthierSteinmanUnsteadyDec           problem_Type;


// Do not edit
int main(int argc, char **argv)
{
    using namespace LifeV;
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    std::cout<< "MPI Initialization\n";
#endif

#ifdef EPETRA_MPI
    std::cout << "Epetra Initialization" << std::endl;
    commPtr_Type commPtr( new Epetra_MpiComm(MPI_COMM_WORLD) );
#else
    commPtr_Type commPtr( new Epetra_SerialComm() );
#endif


    //=======================================================================//
    // Load data from file                                                   //
    //=======================================================================//
    GetPot command_line(argc, argv);
    std::string fileName =
                    command_line.follow("data", 2, "-f", "--file");

    Displayer displayer(commPtr);
    displayer.leaderPrint( "\n[computeVLS] Reading the data... " );
    GetPot getPot = GetPot( fileName );
    displayer.leaderPrint( " done!\n" );

    // Load user-defined parameters for the analytical solution
    problem_Type::setParamsFromGetPot( getPot );

    Real viscosity( getPot("problem/viscosity", 0.035) );

    //=======================================================================//
    // Mesh Stuff                                                            //
    //=======================================================================//
    boost::shared_ptr<mesh_Type> mesh(new mesh_Type);

    MeshData meshData( getPot, "space_discretization" );
    if( getPot("space_discretization/mesh_file", "").compare("") == 0 )
    {
        UInt nElements(getPot("space_discretization/nElements", 20));
        Real xMin( getPot( "space_discretization/xMin", -1. ) );
        Real xMax( getPot( "space_discretization/xMax", 1. ) );
        Real yMin( getPot( "space_discretization/yMin", -1. ) );
        Real yMax( getPot( "space_discretization/yMax", 1. ) );
        Real zMin( getPot( "space_discretization/zMin", -1. ) );
        Real zMax( getPot( "space_discretization/zMax", 1. ) );

        displayer.leaderPrint("\n[computeVLS] Building a regular mesh of ", nElements,
                              " elements per direction\n");
        regularMesh3D(*mesh, 0, nElements, nElements, nElements, false,
                      xMax-xMin, yMax-yMin, zMax-zMin, xMin, yMin, zMin);
        displayer.leaderPrint( "\n\t done!\n" );
    }
    else
    {
        displayer.leaderPrint("\n[computeVLS] Reading mesh from file ", meshData.meshFile(), "\n");
        readMesh(*mesh, meshData);
        displayer.leaderPrint( "\n\t done!\n" );
    }
    MeshPartitioner<mesh_Type> meshPartitioner(mesh, commPtr);
    displayer.leaderPrint( "\n[computeVLS] Mesh acquired and partitioning done. OK\n" );

    //if(verbose)
    //  dataMesh.showMe();
    meshPartitioner.releaseUnpartitionedMesh();
    mesh.reset();

    meshPtr_Type meshPtr = meshPartitioner.meshPartition();


    //=======================================================================//
    // FESpace                                                               //
    //=======================================================================//
    // std::string feOrder = getPot("space_discretization/grad_fespace","P2");

    // feSpacePtr_Type dUFESpacePtr( new feSpace_Type(
    //                 meshPtr, feOrder, nDimensions, commPtr) );

    // using P1 for now
    std::string feOrder = "P1";

    // This is possibly going to be useless in the future. Right now we assume
    // that the solution read from file is a P1 vector field
    feSpacePtr_Type vectorP1FESpacePtr, scalarP1FESpacePtr;
    // if( dUFESpacePtr->refFE().type() == FE_P1bubble_3D )
    vectorP1FESpacePtr.reset(new feSpace_Type(
                    meshPtr, "P1", nDimensions, commPtr));
    scalarP1FESpacePtr.reset(new feSpace_Type( meshPtr, "P1", 1, commPtr));

    std::string displayerString = "\n[computeVLS] FE Spaces created. Using " + feOrder + " elements. OK\n";
    displayer.leaderPrint( displayerString );

    displayer.leaderPrint( "\n[computeVLS] Gradient Dof ", nDimensions*vectorP1FESpacePtr->dim(), "\n" );


    //=======================================================================//
    // Vectors                                                               //
    //=======================================================================//
    std::vector<vectorPtr_Type> dURepeatedPtr, dUExporterPtr;
    vectorPtr_Type frictionURepeatedPtr, frictionUExporterPtr;
    vectorPtr_Type normSRepeatedPtr, normSExporterPtr;
    vectorPtr_Type normSExactUniquePtr, normSExactExporterPtr;
    vectorPtr_Type patchVolumeUniquePtr, patchSizeUniquePtr;
    vectorPtr_Type averageVolumeUniquePtr, averageVolumeRepeatedPtr, averageLengthRepeatedPtr;
    vectorPtr_Type viscousLengthScaleRepeatedPtr, viscousLengthScaleExporterPtr;
    vectorPtr_Type viscousTimeScaleRepeatedPtr, viscousTimeScaleExporterPtr;
    vectorPtr_Type patchVolumeExporterPtr, patchSizeExporterPtr, averageVolumeExporterPtr, averageLengthExporterPtr;
    vectorPtr_Type uImporterPtr, uExporterPtr, uRepeatedPtr, uUniquePtr;

    // the derivative is a vector with fieldDim scalar components
    dURepeatedPtr.resize(nDimensions);
    for( UInt iCoor = 0; iCoor < nDimensions; ++iCoor )
    {
        dURepeatedPtr[iCoor].reset( new vector_Type( vectorP1FESpacePtr->map(), Repeated ) );
    }

    frictionURepeatedPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), Repeated ) );

    normSRepeatedPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), Repeated ) );
    //normSExactRepeatedPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), Repeated ) );
    normSExactUniquePtr.reset( new vector_Type( scalarP1FESpacePtr->map(), Unique ) );

    patchVolumeUniquePtr.reset( new vector_Type( scalarP1FESpacePtr->map(), Unique) );
    patchSizeUniquePtr.reset( new vector_Type( scalarP1FESpacePtr->map(), Unique ) );
    averageVolumeUniquePtr.reset( new vector_Type( scalarP1FESpacePtr->map(), Unique ) );
    averageVolumeRepeatedPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), Repeated ) );
    averageLengthRepeatedPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), Repeated ) );

    viscousLengthScaleRepeatedPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), Repeated ) );
    viscousTimeScaleRepeatedPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), Repeated ) );

    //patchVolumeUniquePtr.reset( new vector_Type( scalarP1FESpacePtr->map(), Unique ) );
    //patchSizeUniquePtr.reset( new vector_Type( scalarP1FESpacePtr->map(), Unique ) );
    //averageVolumeUniquePtr.reset( new vector_Type( scalarP1FESpacePtr->map(), Unique ) );
    //averageLengthRepeatedPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), Unique ) );

    uRepeatedPtr.reset( new vector_Type( vectorP1FESpacePtr->map(), Repeated ) );
    uUniquePtr.reset( new vector_Type( vectorP1FESpacePtr->map(), Unique ) );


    //=======================================================================//
    // Calculator                                                            //
    //=======================================================================//
    displayer.leaderPrint( "\n[computeVLS] Creating a GradientCalculator object...\n" );

    LifeV::GradientCalculator<mesh_Type> gradientCalculator;

    gradientCalculator.setFESpacePtr( vectorP1FESpacePtr );
    gradientCalculator.createLinearSolver( getPot );
    gradientCalculator.updateSystem();

    displayer.leaderPrint( "\n\t done!\n" );


    //=======================================================================//
    // Importer                                                              //
    //=======================================================================//
    displayer.leaderPrint( "\n[computeVLS] Creating the importer...\n" );

    boost::shared_ptr< LifeV::Exporter<mesh_Type> > importerPtr;

    std::string importerType( getPot("importer/type", "none") );
    std::string importerPrefix( getPot("importer/prefix", "importer" ) );

    if (importerType.compare("none") == 0)
        importerPtr.reset( new ExporterEmpty<mesh_Type >
    ( getPot, meshPtr, importerPrefix, commPtr->MyPID()) );
    else
    {
        if (importerType.compare("VTK") == 0)
            importerPtr.reset( new ExporterVTK<mesh_Type > ( getPot, importerPrefix ) );
        else
            importerPtr.reset( new ExporterEnsight<mesh_Type >
        ( getPot, meshPtr, importerPrefix, commPtr->MyPID()) );
    }

    importerPtr->setDataFromGetPot( getPot, "importer" );
    // importerPtr->setPostDir( getPot("importer/post_dir", "./") );
    importerPtr->setMeshProcId( meshPtr, commPtr->MyPID() );

    uImporterPtr.reset( new vector_Type(vectorP1FESpacePtr->map(), importerPtr->mapType() ) );

    importerPtr->addVariable( ExporterData<mesh_Type>::VectorField,
                              getPot("importer/varName", "velocity" ),
                              vectorP1FESpacePtr, uImporterPtr, UInt(0) );

    displayer.leaderPrint( "\n\t done!\n" );


    //=======================================================================//
    // Exporter                                                              //
    //=======================================================================//
    displayer.leaderPrint( "\n[computeVLS] Creating the exporter...\n" );

    boost::shared_ptr< LifeV::Exporter<mesh_Type> > exporterPtr;

    std::string exporterType(getPot("exporter/type", "hdf5"));

#ifdef HAVE_HDF5
    if (exporterType.compare("hdf5") == 0)
        exporterPtr.reset( new ExporterHDF5<mesh_Type > ( getPot, "gradient" ) );
    else
#endif
    {
        if (exporterType.compare("none") == 0)
            exporterPtr.reset( new ExporterEmpty<mesh_Type >
        ( getPot, meshPtr, "gradient", commPtr->MyPID()) );
        else
        {
            if (exporterType.compare("VTK") == 0)
                exporterPtr.reset( new ExporterVTK<mesh_Type > ( getPot, "gradient" ) );
            else
                exporterPtr.reset( new ExporterEnsight<mesh_Type >
            ( getPot, meshPtr, "gradient", commPtr->MyPID()) );
        }
    }

    exporterPtr->setDataFromGetPot( getPot, "exporter" );
    // exporterPtr->setPostDir( getPot("exporter/post_dir", "./") );
    exporterPtr->setMeshProcId( meshPtr, commPtr->MyPID() );

    dUExporterPtr.resize(nDimensions);

    // if( getPot("space_discretization/grad_fespace", "P2") == "P1Bubble" )
    // {
    displayer.leaderPrint( "\n Exporter: needs interpolation P1Bubble -> P1 ...\n");

    for( UInt iCoor = 0; iCoor < nDimensions; ++iCoor )
    {
        std::stringstream varName;

        dUExporterPtr[iCoor].reset( new VectorEpetra(vectorP1FESpacePtr->map(), exporterPtr->mapType() ) );

        varName.str("");
        varName << "dU_dx" << iCoor << "_computed";
        exporterPtr->addVariable( ExporterData<mesh_Type>::VectorField, varName.str(),
                                  vectorP1FESpacePtr, dUExporterPtr[iCoor], UInt(0) );

    }
    // }
    // else
    // {
    //     for( UInt iCoor = 0; iCoor < nDimensions; ++iCoor )
    //     {
    //         std::stringstream varName;

    //         dUExporterPtr[iCoor].reset( new vector_Type( dUFESpacePtr->map(), exporterPtr->mapType() ) );

    //         varName.str("");
    //         varName << "dU_dx" << iCoor << "_computed";
    //         exporterPtr->addVariable( ExporterData<mesh_Type>::VectorField, varName.str(),
    //                                   dUFESpacePtr, dUExporterPtr[iCoor], UInt(0) );

    //     }

    // }

    uExporterPtr.reset( new vector_Type( vectorP1FESpacePtr->map(), exporterPtr->mapType() ) );
    exporterPtr->addVariable( ExporterData<mesh_Type>::VectorField, "u",
                              vectorP1FESpacePtr, uExporterPtr, UInt(0) );

    patchVolumeExporterPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), exporterPtr->mapType() ) );
    exporterPtr->addVariable( ExporterData<mesh_Type>::ScalarField, "patch_volume",
                              scalarP1FESpacePtr, patchVolumeExporterPtr, UInt(0) );
    patchSizeExporterPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), exporterPtr->mapType() ) );
    exporterPtr->addVariable( ExporterData<mesh_Type>::ScalarField, "patch_size",
                              scalarP1FESpacePtr, patchSizeExporterPtr, UInt(0) );
    averageVolumeExporterPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), exporterPtr->mapType() ) );
    exporterPtr->addVariable( ExporterData<mesh_Type>::ScalarField, "average_volume",
                              scalarP1FESpacePtr, averageVolumeExporterPtr, UInt(0) );
    averageLengthExporterPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), exporterPtr->mapType() ) );
    exporterPtr->addVariable( ExporterData<mesh_Type>::ScalarField, "average_length",
                              scalarP1FESpacePtr, averageLengthExporterPtr, UInt(0) );

    viscousLengthScaleExporterPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), exporterPtr->mapType() ) );
    exporterPtr->addVariable( ExporterData<mesh_Type>::ScalarField, "viscous_length_scale",
                              scalarP1FESpacePtr, viscousLengthScaleExporterPtr, UInt(0) );
    viscousTimeScaleExporterPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), exporterPtr->mapType() ) );
    exporterPtr->addVariable( ExporterData<mesh_Type>::ScalarField, "viscous_time_scale",
                              scalarP1FESpacePtr, viscousTimeScaleExporterPtr, UInt(0) );


    frictionUExporterPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), exporterPtr->mapType() ) );
    exporterPtr->addVariable( ExporterData<mesh_Type>::ScalarField, "friction_u",
                              scalarP1FESpacePtr, frictionUExporterPtr, UInt(0) );

    normSExporterPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), exporterPtr->mapType() ) );
    exporterPtr->addVariable( ExporterData<mesh_Type>::ScalarField, "norm_S",
                              scalarP1FESpacePtr, normSExporterPtr, UInt(0) );

    normSExactExporterPtr.reset( new vector_Type( scalarP1FESpacePtr->map(), exporterPtr->mapType() ) );
    exporterPtr->addVariable( ExporterData<mesh_Type>::ScalarField, "norm_S_exact",
                              scalarP1FESpacePtr, normSExactExporterPtr, UInt(0) );

    displayer.leaderPrint( "\n\t done!\n" );


    // ===================================================
    // "offline" operations
    // ===================================================
    //UInt  myScalarDOFs   ( scalarP1FESpacePtr->map().map(Repeated)->NumMyElements() );
    const UInt nbElements( meshPtr->numElements() );
    const UInt nbFEDof   ( scalarP1FESpacePtr->fe().nbFEDof() );
    UInt iGlobalID;

    for( UInt iEl=0; iEl<nbElements; ++iEl )
    {
        scalarP1FESpacePtr->fe().update( meshPtr->element(iEl), UPDATE_WDET );
        Real elementVolume( scalarP1FESpacePtr->fe().measure() );

        // Loop over the nodes
        for (UInt iDof(0); iDof < nbFEDof ; ++iDof)
        {
            iGlobalID = scalarP1FESpacePtr->dof().localToGlobalMap( iEl, iDof );
            patchSizeUniquePtr->sumIntoGlobalValues( iGlobalID, 1. );
            patchVolumeUniquePtr->sumIntoGlobalValues( iGlobalID, elementVolume );
        }
    }
    patchSizeUniquePtr->globalAssemble(Add);
    patchVolumeUniquePtr->globalAssemble(Add);

    // this has the same effect as a GlobalAssemble
    // *patchSizeUniquePtr = *patchSizeUniquePtr;
    // this has the same effect as a GlobalAssemble
    // *patchVolumeUniquePtr = *patchVolumeUniquePtr;

    //! Multiply a Epetra_MultiVector by the reciprocal of another, element-by-element.
    averageVolumeUniquePtr->epetraVector().ReciprocalMultiply( 1.0, patchSizeUniquePtr->epetraVector(),
                                                               patchVolumeUniquePtr->epetraVector(), 0.0 );

    *averageVolumeRepeatedPtr = *averageVolumeUniquePtr;

    for( UInt iEl=0; iEl<nbElements; ++iEl )
    {
        // Loop over the nodes
        for (UInt iDof(0); iDof < nbFEDof ; ++iDof)
        {
            iGlobalID = scalarP1FESpacePtr->dof().localToGlobalMap( iEl, iDof );
            Real volume( averageVolumeRepeatedPtr->operator()( iGlobalID ) );
            Real length( 12. / ( std::sqrt(2.) ) * volume ); length = std::pow(length,1./3.);
            //std::cout << volume << " " << length << " " << std::pow(volume,1./3.) << std::endl;
            averageLengthRepeatedPtr->operator()( iGlobalID ) = length;
        }
    }

    *patchVolumeExporterPtr = *patchVolumeUniquePtr;
    *patchSizeExporterPtr = *patchSizeUniquePtr;
    *averageVolumeExporterPtr = *averageVolumeUniquePtr;
    *averageLengthExporterPtr = *averageLengthRepeatedPtr;

    // ===================================================
    // Start the temporal loop
    // ===================================================
    displayer.leaderPrint( "\n[computeVLS] Loading velocity field and solving... \n" );

    // Temporal parameters
    Real startT = getPot( "time_discretization/initialtime", 0. );
    Real dt     = getPot( "time_discretization/timestep", 1.e-2 );
    Real endT   = getPot( "time_discretization/endtime", 1. );

    UInt timeIndex( static_cast<UInt>( (startT+dt/2) / dt ) );
    // importerPtr->setTimeIndexStart( static_cast<UInt>( timeIndex/importerPtr->save() ) );
    importerPtr->setTimeIndex( timeIndex );
    // importerPtr->import( startT );

    exporterPtr->setTimeIndexStart( static_cast<UInt>( timeIndex/exporterPtr->save() ) );
    exporterPtr->setTimeIndex( timeIndex );
    // exporterPtr->postProcess( startT );

    std::stringstream displayerSString;
    displayerSString << "\n[computeVLS] Initial time index = " << timeIndex;
    displayer.leaderPrint( displayerSString.str() );

    for( Real t = startT; t <= endT+dt/2; t += dt )
    {
        LifeV::LifeChrono chrono;

        displayerSString.str("");
        displayerSString << "\n\n= = = = = We are now at time t = " << t << " s. = = = = = .\n";
        displayer.leaderPrint( displayerSString.str() );

        displayerSString.str("");
        // timeIndex = static_cast<UInt>( (t+dt/2) / dt );
        displayerSString << "\n[computeVLS] Current time index = " << importerPtr->timeIndex();
        displayer.leaderPrint( displayerSString.str() );

        if( getPot("problem/analyticalSolution", false) == false )
        {
            displayerSString.str("");
            //timeIndex = static_cast<UInt>( (t+dt/2) / dt );
            displayerSString << "\n[computeVLS] Importing solution vector...";
            displayer.leaderPrint( displayerSString.str() );

            importerPtr->import( t );
            uUniquePtr->setCombineMode( Zero );
            *uUniquePtr = *uImporterPtr;
        }
        else
        {
            displayerSString.str("");
            //timeIndex = static_cast<UInt>( (t+dt/2) / dt );
            displayerSString << "\n[computeVLS] Interpolating the analytical solution...";
            displayer.leaderPrint( displayerSString.str() );

            vectorP1FESpacePtr->interpolate( problem_Type::uexact, *uUniquePtr, t );
        }
        *uExporterPtr = *uUniquePtr;

        scalarP1FESpacePtr->interpolate( problem_Type::normS, *normSExactUniquePtr, t );
        *normSExactExporterPtr = *normSExactUniquePtr;

        displayerSString.str("");
        Real res(0.);
        uUniquePtr->norm2(&res);
        //timeIndex = static_cast<UInt>( (t+dt/2) / dt );
        displayerSString << "\n[computeVLS] Imported solution has norm = " << res;
        displayer.leaderPrint( displayerSString.str() );

        displayerSString.str("");
        displayerSString << "\n[computeVLS] Computing the velocity gradient... " << std::flush;
        displayer.leaderPrint( displayerSString.str() );
        chrono.start();

        gradientCalculator.computeGradient(*uUniquePtr);

        chrono.stop();
        displayerSString.str("");
        displayerSString << "done in... " << chrono.diff() << std::endl;
        displayer.leaderPrint( displayerSString.str() );

        for( UInt iCoor = 0; iCoor < nDimensions; ++iCoor )
        {
            *dURepeatedPtr[iCoor] = *gradientCalculator.du_dxj( iCoor );
            *dUExporterPtr[iCoor] = *dURepeatedPtr[iCoor];
        }

        // We now have the nodal values of the gradient. We want the nodal values of the viscous
        // length scale, defined as
        // l = ( u* delta(l) / nu )
        // Where delta(l) is a local length scale that represents the averaged extent of the
        // tetrahedron grid cell delta(l) = 1Ú12 sqrt(2) delta(vol)^(1/3), where delta(vol) is
        // the tetrahedron volume. The friction velocity, u*, is given as
        // (u*)^2 = nu ( S_ij S_ij )^(1/2)
        // S_ij = 1/2 ( d u_i / d x_j + d u_j / d x_i )

        UInt dim( vectorP1FESpacePtr->dim() );
        Real normS(0.);

        for( UInt iEl=0; iEl<nbElements; ++iEl )
        {
            vectorP1FESpacePtr->fe().update( meshPtr->element(iEl), UPDATE_WDET );

            // Loop over the nodes
            for (UInt iDof(0); iDof < nbFEDof ; ++iDof)
            {
                normS = 0.;

                iGlobalID = vectorP1FESpacePtr->dof().localToGlobalMap( iEl, iDof );

                // d u_i / d x_j --> dURepeatedPtr[jCoor]->operator()( nTotalDof+iDim + iDof )
                for( UInt iCoor=0; iCoor < nDimensions; ++iCoor )
                {
                    Real u_ii( dURepeatedPtr[iCoor]->operator()(iCoor*dim + iGlobalID) );
                    normS += u_ii * u_ii;
                }
                for( UInt iCoor=0; iCoor < nDimensions; ++iCoor )
                {
                    for( UInt jCoor=iCoor+1; jCoor < nDimensions; ++jCoor )
                    {
                        Real u_ij( dURepeatedPtr[jCoor]->operator()(iCoor*dim + iGlobalID) );
                        Real u_ji( dURepeatedPtr[iCoor]->operator()(jCoor*dim + iGlobalID) );
                        //std::cout << "\nu_ij " << u_ij << std::endl;
                        Real S_ij = 0.5 * ( u_ij + u_ji );
                        normS += 2 * S_ij * S_ij;
                    }
                }
                //if( !normS ) displayer.leaderPrint( "\nZERO NORM\n" );
                normSRepeatedPtr->operator()(iGlobalID) = std::sqrt(normS);
                //normSRepeatedPtr->sumIntoGlobalValues(iGlobalID, std::sqrt(normS));
                frictionURepeatedPtr->operator()(iGlobalID) = viscosity * std::sqrt(normS);
                viscousLengthScaleRepeatedPtr->operator()(iGlobalID) =
                                averageLengthRepeatedPtr->operator()(iGlobalID) * std::sqrt(normS);
                viscousTimeScaleRepeatedPtr->operator()(iGlobalID) =
                                viscosity / frictionURepeatedPtr->operator()(iGlobalID);
            }
        }

        /*
        double * rawDuRepeated[nDimensions], * rawNormSRepeated;
        int myScalarSize, myVectorSize;

        normSRepeatedPtr->epetraVector().PutScalar(0.0);
        normSRepeatedPtr->epetraVector().ExtractView ( &rawNormSRepeated, &myScalarSize );

        std::cout << "\nProc " << commPtr->MyPID() << " has " << myScalarSize << " scalar entries.\n" << std::endl;

        dURepeatedPtr[0]->epetraVector().ExtractView ( &rawDuRepeated[0], &myVectorSize );
        dURepeatedPtr[1]->epetraVector().ExtractView ( &rawDuRepeated[1], &myVectorSize );
        dURepeatedPtr[2]->epetraVector().ExtractView ( &rawDuRepeated[2], &myVectorSize );

        std::cout << "\nProc " << commPtr->MyPID() << " has " << myVectorSize << " vector entries.\n" << std::endl;

        Real normS(0.);
        for( UInt iComp = 0; iComp < myScalarSize; ++iComp )
        {
            // d u_i / d x_j --> dURepeatedPtr[jCoor]->operator()( nTotalDof+iDim + iDof )
            for( UInt iCoor=0; iCoor < nDimensions; ++iCoor )
            {
                Real u_ii( rawDuRepeated[iCoor][iCoor*myScalarSize + iComp] );
                normS += u_ii * u_ii;
            }
            for( UInt iCoor=0; iCoor < nDimensions; ++iCoor )
            {
                for( UInt jCoor=iCoor+1; jCoor < nDimensions; ++jCoor )
                {
                    Real u_ij( rawDuRepeated[jCoor][iCoor*myScalarSize + iComp] );
                    Real u_ji( rawDuRepeated[iCoor][jCoor*myScalarSize + iComp] );
                    //std::cout << "\nu_ij " << u_ij << std::endl;
                    Real S_ij = 0.5 * ( u_ij + u_ji );
                    normS += 2 * S_ij * S_ij;
                }
            }
            //if( !normS ) displayer.leaderPrint( "\nZERO NORM\n" );
            rawNormSRepeated[iComp] = std::sqrt(normS);
        }
         */

        *normSExporterPtr = *normSRepeatedPtr;
        *frictionUExporterPtr = *frictionURepeatedPtr;
        *viscousLengthScaleExporterPtr = *viscousLengthScaleRepeatedPtr;
        *viscousTimeScaleExporterPtr = *viscousTimeScaleRepeatedPtr;

        displayerSString.str("");
        displayerSString << "\n[computeVLS] Writing results on file...\n";
        displayer.leaderPrint( displayerSString.str() );

        exporterPtr->postProcess( t );

        displayerSString.str("");
        displayerSString << " OK\n";
        displayer.leaderPrint( displayerSString.str() );
    }




    commPtr.reset();


#ifdef HAVE_MPI
    MPI_Finalize();
    std::cout<< "MPI Finalization \n";
#endif

    return EXIT_SUCCESS;
}
