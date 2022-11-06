
#ifndef muDetectorConstruction_h
#define muDetectorConstruction_h 1

#include "muAnalyzer.hh"
#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"
#include "G4NistManager.hh" // [yy]
#include "PerlinNoise.hh"

class G4Box;
class G4Orb;
class G4Tubs;
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4Material;

class muDetectorConstruction : public G4VUserDetectorConstruction
{
public:

    muDetectorConstruction();
    ~muDetectorConstruction();

public:

    G4VPhysicalVolume* Construct();

    void UpdateGeometry();

    const G4VPhysicalVolume* GetSensor()     const {return physSensor;};

    void SetAnalyzer(muAnalyzer*);

    void SetVoxelFileName(TString);

private:

    G4Box*             solidSensor;
    G4LogicalVolume*   logicSensor;
    G4VPhysicalVolume* physSensor;
    TString filename;

    G4NistManager* nistMan; // [yy]
    G4Material* EJ200;  // [yy]   Eljen EJ200 (assumed as PVT Scintillator)
    G4Material* Air;    // [yy]
    G4Material* Lead;    // [yy]
    G4Material* GAGG;   // [yy]

    void DefineMaterials();
    muAnalyzer* analyzer;

};

#endif
