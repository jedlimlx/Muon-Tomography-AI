
#ifndef muAnalyzer_hh
#define muAnalyzer_hh

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4Types.hh"

#include <vector>
#include <fstream>
#include <iomanip>

#include "TTree.h"
#include "TFile.h"
#include "TString.h"


class muAnalyzer
{
public:
    muAnalyzer();
    ~muAnalyzer();
    muAnalyzer(const muAnalyzer&);
    
    void SetInit(G4bool, TString);
    
    
    TTree* getTree(){return tree;};
    void Init();
    void Fill(int buf0,                     //nHit [yy] modify
              std::vector<G4int> buf1,   //event [yy] add
              std::vector<G4double> buf2,   //x
              std::vector<G4double> buf3,   //y
              std::vector<G4double> buf4,   //z
              std::vector<G4double> buf5,   //time
              std::vector<G4double> buf6,   //eIn
              std::vector<G4double> buf7,   //eDep
              std::vector<G4int> buf8,   //TrackID
              std::vector<G4int> buf9,   //copyNo
              std::vector<G4int> buf10   //particleID
    );
    void Terminate();
    void SetFileName(TString);
private:
    
    TTree* tree;
    TString filename;
    G4bool isRoot;
    std::ofstream outFile;

    G4int nHit;
    G4int event; // [yy]
    G4double x;
    G4double y;
    G4double z;
    G4double time;
    G4double eIn;
    G4double eDep;
    G4int trackID;
    G4int copyNo;
    G4int particleID;
    
    std::vector<G4int> eventbuf; // [yy]
    std::vector<G4double> xbuf;  // [yy]
    std::vector<G4double> ybuf;  // [yy]
    std::vector<G4double> zbuf;  // [yy]
    std::vector<G4double> timebuf;  // [yy]
    std::vector<G4double> eInbuf;   // [yy]
    std::vector<G4double> eDepbuf;  // [yy]
    std::vector<G4int> trackIDbuf;  // [yy]
    std::vector<G4int> copyNobuf;   // [yy]
    std::vector<G4int> particleIDbuf;  // [yy]
    
};


#endif /* muAnalyzer_hh */
