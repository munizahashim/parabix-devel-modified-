#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: unicode.data

parabix_add_module(
NAME
    unicode.data
SRC
    BidiBrackets.cpp
    BidiMirroring.cpp
    Blocks.cpp
    CaseFolding.cpp
    CompatibilityProperties.cpp
    CompositionExclusions.cpp
    DerivedAge.cpp
    DerivedBidiClass.cpp
    DerivedBinaryProperties.cpp
    DerivedCombiningClass.cpp
    DerivedCoreProperties.cpp
    DerivedDecompositionType.cpp
    DerivedGeneralCategory.cpp
    DerivedJoiningGroup.cpp
    DerivedJoiningType.cpp
    DerivedNormalizationProps.cpp
    DerivedNumericType.cpp
    EastAsianWidth.cpp
    Equivalence.cpp
    GraphemeBreakProperty.cpp
    HangulSyllableType.cpp
    Identity.cpp
    IndicPositionalCategory.cpp
    IndicSyllabicCategory.cpp
    Jamo.cpp
    LineBreak.cpp
    NameAliases.cpp
    PropList.cpp
    PropertyAliases.cpp
    PropertyObjectTable.cpp
    PropertyObjects.cpp
    RadicalSets.cpp
    ScriptExtensions.cpp
    Scripts.cpp
    SentenceBreakProperty.cpp
    SpecialCasing.cpp
    UnicodeData.cpp
    VerticalOrientation.cpp
    WordBreakProperty.cpp
    emoji-data.cpp
DEPS
    unicode.core
    unicode.utf
    re.adt
)
